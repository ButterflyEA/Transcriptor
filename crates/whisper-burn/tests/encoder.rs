use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use whisper_burn::model::encoder::{Conv1d, Encoder};
use whisper_burn::model::ops::{LayerNorm, Linear, gelu_erf};

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

/// Layer norm with unit gain and zero bias (identity when input is already
/// mean-0 / unit-variance along the last dim).
fn identity_ln(size: usize, device: &NdArrayDevice) -> LayerNorm<B> {
    let w: Tensor<B, 1> = Tensor::ones([size], device);
    let b: Tensor<B, 1> = Tensor::zeros([size], device);
    LayerNorm::new(Param::from_tensor(w), Param::from_tensor(b))
}

/// Small encoder built from scrutable parts:
/// - conv1 k=3 s=1 picks the center-aligned input channel (`o` -> channel `o % 2`).
/// - conv2 k=3 s=2 preserves each channel scaled by 0.5.
/// - positional embedding rows [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]].
/// - blocks: identity attention + identity MLP (still normalize via LN).
fn small_encoder(device: &NdArrayDevice) -> Encoder<B> {
    let (n_mels, s, k) = (2usize, 4usize, 3usize);
    let n_head = 2usize;

    // conv1: c1[:, o, :] = x[:, o % 2, :] (center tap only), bias 0.
    let mut w1 = vec![0.0f32; s * n_mels * k];
    for o in 0..s {
        w1[o * (n_mels * k) + (o % n_mels) * k + 1] = 1.0;
    }
    let w1: Tensor<B, 3> = Tensor::from_data(TensorData::new(w1, [s, n_mels, k]), device);
    let conv1 = Conv1d {
        weight: Param::from_tensor(w1),
        bias: Param::from_tensor(Tensor::zeros([s], device)),
    };

    // conv2: c2[:, o, :] = 0.5 * c1[:, o, :] (center tap only), bias 0.
    let mut w2 = vec![0.0f32; s * s * k];
    for o in 0..s {
        w2[o * (s * k) + o * k + 1] = 0.5;
    }
    let w2: Tensor<B, 3> = Tensor::from_data(TensorData::new(w2, [s, s, k]), device);
    let conv2 = Conv1d {
        weight: Param::from_tensor(w2),
        bias: Param::from_tensor(Tensor::zeros([s], device)),
    };

    let pos: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([
            [1.0f32, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ]),
        device,
    );

    let attn = whisper_burn::model::attention::MultiHeadAttention::new(
        identity_linear(s, device),
        identity_linear(s, device),
        identity_linear(s, device),
        identity_linear(s, device),
        n_head,
    );
    let ln = identity_ln(s, device);
    let block = whisper_burn::model::block::ResidualAttentionBlock {
        attn,
        attn_ln: ln.clone(),
        mlp: identity_linear(s, device),
        mlp_ln: ln.clone(),
        mlp2: identity_linear(s, device),
        ln: Some(ln.clone()),
        xa_attn: None,
        xa_attn_ln: None,
        xa_ln: None,
    };

    Encoder {
        conv1,
        conv2,
        positional_embedding: Param::from_tensor(pos),
        blocks: vec![block],
        ln_post: ln,
    }
}

#[test]
fn conv1d_matches_naive_reference() {
    let device = NdArrayDevice::default();
    let xd: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    ];
    let wd: Vec<Vec<Vec<f32>>> = vec![
        vec![
            vec![1.0, 0.0, -1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ],
        vec![
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ],
    ];
    let bd = vec![0.1, -0.2];
    let x: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(xd.iter().flatten().copied().collect(), [1, 3, 8]),
        &device,
    );
    let w: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(wd.iter().flatten().flatten().copied().collect(), [2, 3, 3]),
        &device,
    );
    let bias: Tensor<B, 1> = Tensor::from_data(TensorData::new(bd.clone(), [2]), &device);
    let conv = Conv1d {
        weight: Param::from_tensor(w),
        bias: Param::from_tensor(bias),
    };
    for stride in [1usize, 2usize] {
        let got = conv
            .forward(x.clone(), stride)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (nout, k) = (wd.len(), wd[0][0].len());
        let (c, l) = (xd.len(), xd[0].len());
        let lp = l + 2; // pad = (k-1)/2 = 1
        let n_windows = lp - k + 1;
        let out_len = n_windows / stride;
        let mut exp = vec![0.0f32; nout * out_len];
        for o in 0..nout {
            for t in 0..out_len {
                let mut acc = bd[o];
                for ci in 0..c {
                    for kk in 0..k {
                        let xp = if t * stride + kk >= 1 && t * stride + kk <= l {
                            xd[ci][t * stride + kk - 1]
                        } else {
                            0.0
                        };
                        acc += xp * wd[o][ci][kk];
                    }
                }
                exp[o * out_len + t] = acc;
            }
        }
        assert_eq!(got.len(), exp.len(), "stride {stride}");
        for (a, b) in got.iter().zip(exp.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "stride {stride}: got {got:?}\nexp {exp:?}"
            );
        }
    }
}

#[test]
fn encoder_forward_matches_manual_pipeline() {
    // Pins the encoder wiring against the reference layer order:
    //   gelu(conv1(x,1)) -> gelu(conv2(_,2)) -> swap(1,2) -> +pos -> blocks -> ln_post.
    // Each component (conv, gelu, block, ln) is independently golden-tested, so
    // divergences here (wrong stride, missing gelu, no pos add, wrong LN, ...)
    // are exactly the mistakes this test can detect.
    let device = NdArrayDevice::default();
    let n_mels = 2usize;
    let s = 4usize;
    let enc = small_encoder(&device);

    let x: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(
            vec![
                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 1.0, 0.5, 0.25, 2.0, 1.0, 0.5,
                0.25,
            ],
            [1, n_mels, 8],
        ),
        &device,
    );

    let got = enc.forward(x.clone());
    assert_eq!(got.shape().dims(), [1, 4, s]);

    let mut y = gelu_erf(enc.conv1.forward(x, 1));
    y = gelu_erf(enc.conv2.forward(y, 2));
    y = y
        .swap_dims(1, 2)
        .add(enc.positional_embedding.val().unsqueeze::<3>());
    for blk in &enc.blocks {
        y = blk.forward(y, None, None);
    }
    let expected = enc.ln_post.forward(y);

    let g = got.into_data().to_vec::<f32>().unwrap();
    let e = expected.into_data().to_vec::<f32>().unwrap();
    assert_eq!(g.len(), e.len());
    for (a, b) in g.iter().zip(e.iter()) {
        assert!((a - b).abs() < 1e-5, "got {g:?}\nexp {e:?}");
    }
    // Non-degenerate: values differ across timesteps and features.
    assert!(
        (g[0] - g[4]).abs() > 1e-5,
        "output should vary across time: {g:?}"
    );
    assert!(
        (g[0] - g[1]).abs() > 1e-5,
        "output should vary across features: {g:?}"
    );
}
