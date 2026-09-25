use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use whisper_burn::model::attention::MultiHeadAttention;
use whisper_burn::model::block::ResidualAttentionBlock;
use whisper_burn::model::ops::{LayerNorm, Linear};

type B = NdArray<f32>;

/// Identity linear: weight = I, bias = 0, so forward(x) == x.
fn identity_linear(size: usize, device: &NdArrayDevice) -> Linear<B> {
    let w: Tensor<B, 2> = Tensor::eye(size, device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    Linear {
        weight: Param::from_tensor(w),
        bias: Some(Param::from_tensor(b)),
    }
}

/// Zero linear: weight = 0, bias = 0, so forward(x) == 0.
fn zero_linear(size: usize, device: &NdArrayDevice) -> Linear<B> {
    let w: Tensor<B, 2> = Tensor::zeros([size, size], device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    Linear {
        weight: Param::from_tensor(w),
        bias: Some(Param::from_tensor(b)),
    }
}

/// Layer norm with unit gain and zero bias. For an already mean-0 / unit-variance
/// input this is the identity to within eps (1e-5 relative), e.g. [1, -1] -> [1, -1].
fn identity_ln(size: usize, device: &NdArrayDevice) -> LayerNorm<B> {
    let w: Tensor<B, 1> = Tensor::ones([size], device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    LayerNorm::new(Param::from_tensor(w), Param::from_tensor(b))
}

/// Layer norm with zero gain, so forward(x) == 0 regardless of x.
fn zero_ln(size: usize, device: &NdArrayDevice) -> LayerNorm<B> {
    let w: Tensor<B, 1> = Tensor::zeros([size], device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    LayerNorm::new(Param::from_tensor(w), Param::from_tensor(b))
}

#[test]
fn encoder_block_identity_golden() {
    // Encoder block, single token x = [1, -1] (mean 0, unit variance), so the
    // identity LayerNorms are near-identity. All Linears are identity.
    //   attn_ln(x) ~ [1,-1];   single-key self-attn ctx = v = attn_ln(x);  out = id.
    //   x1 = x + attn = [2,-2].
    //   mlp_ln(x1) ~ [1,-1] (unit-normalized again); mlp = id; gelu([1,-1])
    //     = [0.84134, -0.15866]; mlp2 = id.
    //   y = x1 + gelu(mlp_ln(x1)) = [2.84134, -2.15866].
    let device = NdArrayDevice::default();
    let attn = MultiHeadAttention::new(
        identity_linear(2, &device),
        identity_linear(2, &device),
        identity_linear(2, &device),
        identity_linear(2, &device),
        1,
    );
    let ln = identity_ln(2, &device);
    let block = ResidualAttentionBlock {
        attn,
        attn_ln: identity_ln(2, &device),
        mlp: identity_linear(2, &device),
        mlp_ln: identity_ln(2, &device),
        mlp2: identity_linear(2, &device),
        ln: Some(ln),
        xa_attn: None,
        xa_attn_ln: None,
        xa_ln: None,
    };
    let x: Tensor<B, 3> = Tensor::from_data(TensorData::from([[[1.0f32, -1.0]]]), &device);
    let y = block.forward(x, None, None);
    let got = y.into_data().to_vec::<f32>().unwrap();
    let exp = [2.84134f32, -2.15866];
    for (g, e) in got.iter().zip(exp.iter()) {
        assert!((g - e).abs() < 1e-3, "got {got:?}, expected {exp:?}");
    }
}

#[test]
fn decoder_cross_attention_uses_query_ln_and_raw_kv() {
    // Decoder block with zeroed self-attn and MLP so only the cross path acts.
    //   x = [1,0] (single query); xa = [[2,0],[0,0]] (raw encoder output, NOT normed).
    //   q = xa_attn_ln(x) ~ [1,-1]; kv = xa raw (identity linears).
    //   scores = [1,-1]·[[2,0],[0,0]]ᵀ * 1/√2 = [1.41421, 0]
    //   softmax([1.41421, 0]) = [0.80458, 0.19542]
    //   ctx = 0.80458*[2,0] + 0.19542*[0,0] = [1.60916, 0]
    //   y = x + [1.60916, 0] = [2.60916, 0].
    // This pins that the query is the LN'd x while KV comes from raw xa.
    let device = NdArrayDevice::default();
    let zero = zero_linear(2, &device);
    let attn = MultiHeadAttention::new(zero.clone(), zero.clone(), zero.clone(), zero.clone(), 1);
    let xa_attn = MultiHeadAttention::new(
        identity_linear(2, &device),
        identity_linear(2, &device),
        identity_linear(2, &device),
        identity_linear(2, &device),
        1,
    );
    let ln = identity_ln(2, &device);
    let block = ResidualAttentionBlock {
        attn,
        attn_ln: zero_ln(2, &device),
        mlp: zero.clone(),
        mlp_ln: zero_ln(2, &device),
        mlp2: zero,
        ln: Some(ln),
        xa_attn: Some(xa_attn),
        xa_attn_ln: Some(identity_ln(2, &device)),
        xa_ln: None,
    };
    let x: Tensor<B, 3> = Tensor::from_data(TensorData::from([[[1.0f32, 0.0]]]), &device);
    let xa: Tensor<B, 3> =
        Tensor::from_data(TensorData::from([[[2.0f32, 0.0], [0.0, 0.0]]]), &device);
    let y = block.forward(x, Some(xa), None);
    let got = y.into_data().to_vec::<f32>().unwrap();
    assert!((got[0] - 2.60916).abs() < 5e-3, "got {got:?}");
    assert!(got[1].abs() < 1e-3, "got {got:?}");
}
