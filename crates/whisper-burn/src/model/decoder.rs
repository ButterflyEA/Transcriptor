use crate::model::block::ResidualAttentionBlock;
use crate::model::ops::LayerNorm;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Whisper's causal mask: `torch.full((n, n), -inf)` then `triu_(1)`, so strictly
/// upper-triangular entries are `-inf` and the diagonal/lower triangle are `0`.
/// Broadcast shape `[1, 1, seq, seq]` for `q`/`k` attention scores `[b, h, seq, seq]`.
pub fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut m = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            if c > r {
                m[r * n + c] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_data(TensorData::new(m, [1, 1, n, n]), device)
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    /// `[n_vocab, n_state]`; logits are tied to this embedding.
    pub token_embedding: Param<Tensor<B, 2>>,
    /// `[n_text_ctx, n_state]`.
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln: LayerNorm<B>,
}

impl<B: Backend> Decoder<B> {
    /// `tokens [seq]` -> logits `[seq, n_vocab]` (tied to `token_embedding`).
    pub fn forward(
        &self,
        tokens: Tensor<B, 1, burn::tensor::Int>,
        xa: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [seq] = tokens.shape().dims();
        let [_, n_state] = self.token_embedding.val().shape().dims();
        let emb = self.token_embedding.val().select(0, tokens); // [seq, n_state]
        let pos = self.positional_embedding.val().slice([0..seq, 0..n_state]);
        let mut x = emb.add(pos).unsqueeze::<3>(); // [1, seq, n_state]
        let mask = causal_mask::<B>(seq, &xa.device());
        for block in &self.blocks {
            x = block.forward(x, Some(xa.clone()), Some(&mask));
        }
        let x = self.ln.forward(x); // [1, seq, n_state]
        let w = self.token_embedding.val().transpose(); // [n_state, n_vocab]
        x.squeeze_dim::<2>(0).matmul(w)
    }
}
