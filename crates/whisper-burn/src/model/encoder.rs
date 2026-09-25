use crate::model::block::ResidualAttentionBlock;
use crate::model::ops::{LayerNorm, gelu_erf};
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// 1-D same-pad convolution over `[b, c, l]` with weight `[out, c, k]`.
/// Matches whisper's `nn.Conv1d(out, in, 3, padding=1)` behavior.
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Conv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>, stride: usize) -> Tensor<B, 3> {
        let [b, c, l] = x.shape().dims();
        let device = x.device();
        let [out, c2, k] = self.weight.val().shape().dims();
        assert_eq!(c, c2);
        let pad = (k - 1) / 2;
        // zero-pad symmetric
        let left: Tensor<B, 3> = Tensor::zeros([b, c, pad], &device);
        let right: Tensor<B, 3> = Tensor::zeros([b, c, pad], &device);
        let xp = Tensor::cat(vec![left, x, right], 2); // [b, c, l + 2p]
        let xp = xp.swap_dims(1, 2); // [b, L+2p, c]
        let lp = l + 2 * pad;
        let out_len = lp - k + 1;
        let mut acc: Option<Tensor<B, 3>> = None;
        for kk in 0..k {
            let w_k = self
                .weight
                .val()
                .slice([0..out, 0..c, kk..kk + 1])
                .reshape([out, c]);
            let x_k = xp.clone().slice([0..b, kk..kk + out_len, 0..c]); // [b, out_len, c]
            let contrib = x_k.matmul(w_k.transpose().unsqueeze::<3>()); // [b, out_len, out]
            acc = Some(match acc {
                Some(a) => a.add(contrib),
                None => contrib,
            });
        }
        let mut y = acc.unwrap().swap_dims(1, 2); // [b, out, out_len]
        if stride == 2 {
            // keep even starts: select columns 0,2,4,... via a binary matmul
            let half = out_len / 2;
            let mut sel = vec![0.0f32; out_len * half];
            for t in 0..half {
                sel[(2 * t) * half + t] = 1.0;
            }
            let e: Tensor<B, 2> = Tensor::from_data(TensorData::new(sel, [out_len, half]), &device);
            y = y.matmul(e.unsqueeze::<3>());
        } else {
            assert_eq!(stride, 1);
        }
        y.add(self.bias.val().reshape([1, out, 1]))
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln_post: LayerNorm<B>,
}

impl<B: Backend> Encoder<B> {
    /// mel `[1, n_mels, 3000]` -> features `[1, 1500, n_state]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = gelu_erf(self.conv1.forward(x, 1)); // [1, n_state, 3000]
        x = gelu_erf(self.conv2.forward(x, 2)); // [1, n_state, 1500]
        x = x
            .swap_dims(1, 2)
            .add(self.positional_embedding.val().unsqueeze::<3>()); // [1, 1500, n_state]
        for block in &self.blocks {
            x = block.forward(x, None, None);
        }
        self.ln_post.forward(x)
    }
}
