use super::attention::MultiHeadAttention;
use super::ops::{LayerNorm, Linear, gelu_erf};
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ResidualAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub attn_ln: LayerNorm<B>,
    pub mlp: Linear<B>,
    pub mlp_ln: LayerNorm<B>,
    pub mlp2: Linear<B>,
    /// Present in older OpenAI `.pt` checkpoints (`{base}.ln.weight/bias`) but
    /// never used in forward; absent from HF-format checkpoints. Kept so a
    /// strict `WeightMap::finish` drains it when present.
    pub ln: Option<LayerNorm<B>>,
    pub xa_attn: Option<MultiHeadAttention<B>>,
    pub xa_attn_ln: Option<LayerNorm<B>>,
    /// Present in some OpenAI `.pt` checkpoints (`decoder.blocks.{i}.xa_ln.weight`,
    /// bias-less) but never used in forward; kept so `WeightMap::finish` drains it.
    pub xa_ln: Option<Linear<B>>,
}

impl<B: Backend> ResidualAttentionBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Option<Tensor<B, 3>>,
        mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let att = self
            .attn
            .forward(self.attn_ln.forward(x.clone()), None, mask);
        let x = x.add(att);
        let x = match (xa, &self.xa_attn, &self.xa_attn_ln) {
            (Some(xa), Some(xa_attn), Some(xa_attn_ln)) => {
                let q = xa_attn_ln.forward(x.clone());
                x.add(xa_attn.forward(q, Some(xa), None))
            }
            _ => x,
        };
        let m = self.mlp.forward(self.mlp_ln.forward(x.clone()));
        let act = gelu_erf(m);
        x.add(self.mlp2.forward(act))
    }
}
