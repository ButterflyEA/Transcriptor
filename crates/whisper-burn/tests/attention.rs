use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use whisper_burn::model::attention::MultiHeadAttention;
use whisper_burn::model::ops::Linear;

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

#[test]
fn self_attention_matches_hand_computation() {
    let device = NdArrayDevice::default();
    let l = identity_linear(2, &device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    // x = [[1,0],[0,2]] ; q=k=v=x ; d_head=2 ; scale = 1/sqrt(2)
    // scores = q·kᵀ * scale = [[1,0],[0,4]] / 1.41421
    // row0: softmax([0.70711, 0]) = [0.66977, 0.33023]
    //   out = .66977*[1,0] + .33023*[0,2] = [0.66977, 0.66046]
    // row1: softmax([0, 2.82843]) = [0.0558, 0.9442]
    //   out = .0558*[1,0] + .9442*[0,2] = [0.0558, 1.8884]
    let x: Tensor<B, 3> =
        Tensor::from_data(TensorData::from([[[1.0f32, 0.0], [0.0, 2.0]]]), &device);
    let y = mha.forward(x, None, None);
    let d = y.into_data().to_vec::<f32>().unwrap();
    let exp = [0.66977f32, 0.66046, 0.0558, 1.8884];
    for (g, e) in d.iter().zip(exp.iter()) {
        assert!((g - e).abs() < 5e-3, "got {g}, expected {e}");
    }
}

#[test]
fn cross_attention_uses_xa() {
    let device = NdArrayDevice::default();
    let l = identity_linear(2, &device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    // q = [[1,0]] ; kv = [[2,0],[0,0]] ; d_head=2 ; scale = 1/sqrt(2)
    // scores = q·kᵀ * scale = [2, 0] / 1.41421 = [1.41421, 0]
    // softmax([1.41421, 0]) = [0.80458, 0.19542]
    // out = .80458*[2,0] + .19542*[0,0] = [1.60916, 0]
    let q: Tensor<B, 3> = Tensor::from_data(TensorData::from([[[1.0f32, 0.0]]]), &device);
    let kv: Tensor<B, 3> =
        Tensor::from_data(TensorData::from([[[2.0f32, 0.0], [0.0, 0.0]]]), &device);
    let y = mha.forward(q, Some(kv), None);
    let d = y.into_data().to_vec::<f32>().unwrap();
    assert!((d[0] - 1.60916).abs() < 5e-3, "got {:?}", d);
    assert!(d[1].abs() < 1e-3, "got {:?}", d);
}

#[test]
fn multi_head_splits_and_concats() {
    let device = NdArrayDevice::default();
    let l = identity_linear(4, &device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 2);
    // x = [[1,0,1,0],[0,2,0,3]] ; n_state=4, n_head=2, d_head=2.
    // head0 operates on cols [0,1], head1 on cols [2,3]. Columns differ → distinct maths.
    // head0 tokens: [[1,0],[0,2]] (as in self test)
    //   token0 out = [0.66977, 0.66046]
    //   token1 out = [0.0558, 1.8884]
    // head1 tokens: [[1,0],[0,3]]
    //   scores row0: [0.70711, 0] → softmax [0.66977, 0.33023]
    //     out = .66977*[1,0] + .33023*[0,3] = [0.66977, 0.99069]
    //   scores row1: [0, 9/1.41421=6.36396] → softmax [exp(0),exp(6.364)]/(1+e^6.364)
    //     = [0.00172, 0.99828] → out = .00172*[1,0]+.99828*[0,3] = [0.00172, 2.99484]
    let x: Tensor<B, 3> = Tensor::from_data(
        TensorData::from([[[1.0f32, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 3.0]]]),
        &device,
    );
    let y = mha.forward(x, None, None);
    let d = y.into_data().to_vec::<f32>().unwrap();
    let exp = [
        0.66977f32, 0.66046, 0.66977, 0.99069, // token0 (head0 | head1)
        0.0558, 1.8884, 0.00172, 2.99484, // token1 (head0 | head1)
    ];
    for (g, e) in d.iter().zip(exp.iter()) {
        assert!((g - e).abs() < 5e-3, "got {g}, expected {e}");
    }
}

#[test]
fn causal_mask_restricts_future_keys() {
    let device = NdArrayDevice::default();
    let l = identity_linear(2, &device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    // x = [[2,0],[3,4]] ; identical q,k,v ; causal mask [1,1,2,2] zeroes the upper triangle.
    let x: Tensor<B, 3> =
        Tensor::from_data(TensorData::from([[[2.0f32, 0.0], [3.0, 4.0]]]), &device);
    let mask: Tensor<B, 4> = Tensor::from_data(
        TensorData::from([[[[0.0f32, f32::NEG_INFINITY], [0.0, 0.0]]]]),
        &device,
    );
    let y = mha.forward(x.clone(), None, Some(&mask));
    // row0 may only attend to col0 → out = [2,0]. row1 may attend to col0,col1:
    //   scores row1 = [6, 25]/1.41421 = [4.24264, 17.67767]
    //   softmax ≈ [e^4.24, e^17.68]/(e^4.24+e^17.68) = [~1.5e-6, ~1.0]
    //   out ≈ 1.5e-6*[2,0] + 1.0*[3,4] = [3.000003, 4]
    let d = y.into_data().to_vec::<f32>().unwrap();
    assert!((d[0] - 2.0).abs() < 1e-3, "got {:?}", d);
    assert!(d[1].abs() < 1e-3, "got {:?}", d);
    assert!((d[2] - 3.0).abs() < 1e-3, "got {:?}", d);
    assert!((d[3] - 4.0).abs() < 1e-3, "got {:?}", d);
}
