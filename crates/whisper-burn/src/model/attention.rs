use super::ops::Linear;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub out: Linear<B>,
    pub n_head: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        query: Linear<B>,
        key: Linear<B>,
        value: Linear<B>,
        out: Linear<B>,
        n_head: usize,
    ) -> Self {
        Self {
            query,
            key,
            value,
            out,
            n_head,
        }
    }

    fn qkv(
        &self,
        x: &Tensor<B, 3>,
        xa: Option<&Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let xa = xa.unwrap_or(x);
        (
            self.query.forward(x.clone()),
            self.key.forward(xa.clone()),
            self.value.forward(xa.clone()),
        )
    }

    /// Scaled dot-product attention.
    /// - self-attention when `xa` is `None` (q,k,v all from `x`).
    /// - cross-attention when `xa` is `Some` (k,v from `xa`, q from `x`).
    /// - `mask` (shape `[1, 1, q, k]`) is added to the scores; `-inf` forbids cells.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Option<Tensor<B, 3>>,
        mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(&x, xa.as_ref());
        let [b, m, chan] = q.shape().dims();
        let [_, k_len, _] = k.shape().dims();
        let head = chan / self.n_head;

        let q = q.reshape([b, m, self.n_head, head]).swap_dims(1, 2); // b,h,m,d
        let k = k.reshape([b, k_len, self.n_head, head]).swap_dims(1, 2); // b,h,l,d
        let v = v.reshape([b, k_len, self.n_head, head]).swap_dims(1, 2);

        // scores: b,h,m,l  -- scale folded into kᵀ (whisper applies 1/√d on k).
        let scale = (head as f32).sqrt().recip();
        let mut scores = q.matmul(k.transpose().mul_scalar(scale));
        if let Some(mk) = mask {
            scores = scores.add(mk.clone());
        }
        let attn = burn::tensor::activation::softmax(scores, 3); // softmax over l
        let ctx = attn.matmul(v); // b,h,m,d
        let ctx = ctx.swap_dims(1, 2).reshape([b, m, chan]); // b,m,chan
        self.out.forward(ctx)
    }
}
