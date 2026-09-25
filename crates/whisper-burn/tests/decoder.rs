use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use whisper_burn::model::attention::MultiHeadAttention;
use whisper_burn::model::block::ResidualAttentionBlock;
use whisper_burn::model::decoder::{Decoder, causal_mask};
use whisper_burn::model::ops::{LayerNorm, Linear};

type B = NdArray<f32>;

/// Linear with identity weight and zero bias.
fn identity_linear(size: usize, device: &NdArrayDevice) -> Linear<B> {
    let w: Tensor<B, 2> = Tensor::eye(size, device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    Linear {
        weight: Param::from_tensor(w),
        bias: Some(Param::from_tensor(b)),
    }
}

/// Linear with zero weight and zero bias.
fn zero_linear(size: usize, device: &NdArrayDevice) -> Linear<B> {
    let w: Tensor<B, 2> = Tensor::zeros([size, size], device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    Linear {
        weight: Param::from_tensor(w),
        bias: Some(Param::from_tensor(b)),
    }
}

/// Layer norm with unit gain and zero bias.
fn identity_ln(size: usize, device: &NdArrayDevice) -> LayerNorm<B> {
    LayerNorm::new(
        Param::from_tensor(Tensor::ones([size], device)),
        Param::from_tensor(Tensor::zeros([size], device)),
    )
}

fn zero_block(s: usize, n_head: usize, device: &NdArrayDevice) -> ResidualAttentionBlock<B> {
    let mha = || {
        MultiHeadAttention::new(
            zero_linear(s, device),
            zero_linear(s, device),
            zero_linear(s, device),
            zero_linear(s, device),
            n_head,
        )
    };
    let ln = identity_ln(s, device);
    ResidualAttentionBlock {
        attn: mha(),
        attn_ln: ln.clone(),
        mlp: zero_linear(s, device),
        mlp_ln: ln.clone(),
        mlp2: zero_linear(s, device),
        ln: Some(ln.clone()),
        xa_attn: Some(mha()),
        xa_attn_ln: Some(ln.clone()),
        xa_ln: None,
    }
}

/// token_embedding = identity on the first `s` vocab rows, zero-padded after.
fn identity_padded_embedding(n_vocab: usize, s: usize, device: &NdArrayDevice) -> Tensor<B, 2> {
    let mut v = vec![0.0f32; n_vocab * s];
    for i in 0..s {
        v[i * s + i] = 1.0;
    }
    Tensor::from_data(TensorData::new(v, [n_vocab, s]), device)
}

#[test]
fn causal_mask_matches_whisper_convention() {
    // whisper: mask = full((n,n), -inf); mask.triu_(1)  =>  strictly-below diagonal 0, rest -inf.
    let device = NdArrayDevice::default();
    let m = causal_mask::<B>(3, &device);
    assert_eq!(m.shape().dims(), [1, 1, 3, 3]);
    let d = m.into_data().to_vec::<f32>().unwrap();
    let expect = [
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        0.0,
    ];
    for (a, b) in d.iter().zip(expect.iter()) {
        assert_eq!(*a, *b, "got {d:?}");
    }
}

#[test]
fn decoder_zero_weights_matches_tied_manual() {
    // Zero-weight blocks pass x through unchanged (all projections/bias are 0), so
    //   forward(tokens) = ln( embed[tokens] + 0 ) @ token_embedding^T  (tied).
    // With token_embedding = identity-padded and final ln = identity LN, logits row t is
    //   ln(e_t) @ te^T, where e_t is the standard basis row. Identity LN of a basis row
    //   is NOT itself (it is mean-0 / unit-variance over the last dim), so:
    //     ln(e_t)[t] = (s-1)/sqrt(s-1) = sqrt(s-1)
    //     ln(e_t)[i!=t] = -1/sqrt(s-1)
    //     vocab cols beyond s -> 0 (zero-padded rows of te)
    let device = NdArrayDevice::default();
    let (s, n_vocab, n_head, n_text_ctx) = (4usize, 6usize, 2usize, 8usize);

    let te = identity_padded_embedding(n_vocab, s, &device);
    let dec = Decoder {
        token_embedding: Param::from_tensor(te.clone()),
        positional_embedding: Param::from_tensor(Tensor::zeros([n_text_ctx, s], &device)),
        blocks: vec![zero_block(s, n_head, &device)],
        ln: identity_ln(s, &device),
    };

    let tokens = Tensor::<B, 1>::from_data(TensorData::from([0.0f32, 1.0]), &device).int();
    let xa: Tensor<B, 3> = Tensor::zeros([1, 8, s], &device);
    let logits = dec.forward(tokens, xa);
    assert_eq!(logits.shape().dims(), [2, n_vocab]);
    let v = logits.into_data().to_vec::<f32>().unwrap();

    let root = (s as f32 - 1.0).sqrt();
    let off = -1.0 / (s as f32 - 1.0).sqrt();
    for t in 0..2usize {
        for j in 0..n_vocab {
            let expected = if j < s {
                if j == t { root } else { off }
            } else {
                0.0
            };
            assert!(
                (v[t * n_vocab + j] - expected).abs() < 1e-4,
                "row {t} col {j}: got {:?} exp {expected}",
                v[t * n_vocab + j]
            );
        }
    }
}

#[test]
fn decoder_uses_cross_attention_and_causal_mask() {
    // Same decoder but with non-zero (identity) attention/MLP so the causal mask and the
    // cross-attention source actually change the numbers. Reference composes the forward
    // recipe from the decoder's own components; sensitivity asserts prove xa and mask are
    // really consumed (dropping either would change the output).
    let device = NdArrayDevice::default();
    let (s, n_vocab, n_head) = (4usize, 8usize, 2usize);

    let mha = || {
        MultiHeadAttention::new(
            identity_linear(s, &device),
            identity_linear(s, &device),
            identity_linear(s, &device),
            identity_linear(s, &device),
            n_head,
        )
    };
    let ln = identity_ln(s, &device);
    let block = ResidualAttentionBlock {
        attn: mha(),
        attn_ln: ln.clone(),
        mlp: identity_linear(s, &device),
        mlp_ln: ln.clone(),
        mlp2: identity_linear(s, &device),
        ln: Some(ln.clone()),
        xa_attn: Some(mha()),
        xa_attn_ln: Some(ln.clone()),
        xa_ln: None,
    };
    let dec = Decoder {
        token_embedding: Param::from_tensor(identity_padded_embedding(n_vocab, s, &device)),
        positional_embedding: Param::from_tensor(Tensor::from_data(
            TensorData::from([
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]),
            &device,
        )),
        blocks: vec![block],
        ln,
    };

    let seq = 3usize;
    let tokens = Tensor::<B, 1>::from_data(TensorData::from([0.0f32, 1.0, 2.0]), &device).int();
    let xa: Tensor<B, 3> = Tensor::<B, 2>::from_data(
        TensorData::from([
            [0.1f32, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]),
        &device,
    )
    .unsqueeze::<3>();

    let got = dec.forward(tokens.clone(), xa.clone());
    assert_eq!(got.shape().dims(), [seq, n_vocab]);

    // Reference: embed -> +pos -> batch -> blocks(xa, causal mask) -> ln -> tied logits.
    let emb = dec.token_embedding.val().select(0, tokens.clone());
    let p = dec.positional_embedding.val().slice([0..seq, 0..s]);
    let mut y = emb.add(p).unsqueeze::<3>();
    let mask = causal_mask::<B>(seq, &device);
    for blk in &dec.blocks {
        y = blk.forward(y, Some(xa.clone()), Some(&mask));
    }
    y = dec.ln.forward(y);
    let expected = y
        .squeeze_dim::<2>(0)
        .matmul(dec.token_embedding.val().transpose());

    let g = got.into_data().to_vec::<f32>().unwrap();
    let e = expected.into_data().to_vec::<f32>().unwrap();
    assert_eq!(g.len(), e.len());
    for (a, b) in g.iter().zip(e.iter()) {
        assert!((a - b).abs() < 1e-5, "got {g:?}\nexp {e:?}");
    }

    // Sensitivity 1: swapping in a different cross-attention source changes the output.
    let xa_zero: Tensor<B, 3> = Tensor::zeros([1, seq, s], &device);
    let gz = dec
        .forward(tokens.clone(), xa_zero)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let max_d = g
        .iter()
        .zip(gz.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_d > 1e-4, "cross-attention source must matter: {g:?}");

    // Sensitivity 2: dropping the causal mask changes the output.
    let mut y2 = dec
        .token_embedding
        .val()
        .select(0, tokens)
        .add(dec.positional_embedding.val().slice([0..seq, 0..s]))
        .unsqueeze::<3>();
    for blk in &dec.blocks {
        y2 = blk.forward(y2, Some(xa.clone()), None);
    }
    let no_mask = dec
        .ln
        .forward(y2)
        .squeeze_dim::<2>(0)
        .matmul(dec.token_embedding.val().transpose())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let max_d2 = g
        .iter()
        .zip(no_mask.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_d2 > 1e-4, "causal mask must matter: {g:?}");
    // The reference with a mask must match the (already validated) reference above; sanity:
    assert!((g[0] - e[0]).abs() < 1e-5);
}
