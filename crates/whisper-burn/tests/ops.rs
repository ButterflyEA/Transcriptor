use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use whisper_burn::model::ops::{LayerNorm, Linear, erf, gelu_erf};

type B = NdArray<f32>;

#[test]
fn erf_matches_reference_values() {
    for (x, exp) in [(0.0, 0.0), (0.5, 0.520_499_9), (1.0, 0.842_700_8)] {
        assert!(
            (erf(x) - exp).abs() < 1e-5,
            "erf({x}) = {} (expected {exp})",
            erf(x)
        );
    }
}

#[test]
fn gelu_reflects_erf_form() {
    // 0.5 * 1 * (1 + erf(1/sqrt2)) = 0.84134...
    let device = NdArrayDevice::default();
    let x: Tensor<B, 3> = Tensor::from_data(TensorData::from([[[1.0f32]]]), &device);
    let y = gelu_erf(x);
    let v = y.into_data().to_vec::<f32>().unwrap()[0];
    assert!((v - 0.84134).abs() < 1e-4);
}

#[test]
fn linear_forward_matches_manual() {
    let device = NdArrayDevice::default();
    let w: Tensor<B, 2> = Tensor::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let b: Tensor<B, 1> = Tensor::from_data(TensorData::from([0.5f32, -0.5]), &device);
    let lin = Linear {
        weight: burn::module::Param::from_tensor(w),
        bias: Some(burn::module::Param::from_tensor(b)),
    };
    let x: Tensor<B, 3> =
        Tensor::from_data(TensorData::from([[[1.0f32, 1.0], [2.0, 0.0]]]), &device);
    let y = lin.forward(x);
    // Linear: y = x @ Wᵀ + b. Wᵀ = [[1,3],[2,4]]
    // row0: x@Wᵀ = [1*1+1*2, 1*3+1*4] = [3,7] + b = [3.5, 6.5]
    // row1: [2*1+0*2, 2*3+0*4] = [2,6] + b = [2.5, 5.5]
    assert_eq!(
        y.into_data(),
        TensorData::from([[[3.5f32, 6.5], [2.5, 5.5]]])
    );
}

#[test]
fn layer_norm_matches_manual() {
    // LayerNorm over last dim of [1,2,3]: B=1, T=2, C=3
    let device = NdArrayDevice::default();
    let weight = Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::from([1.0f32, 1.0, 1.0]),
        &device,
    ));
    let bias = Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::from([0.0f32, 0.0, 0.0]),
        &device,
    ));
    let ln = LayerNorm::new(weight, bias);
    let x: Tensor<B, 3> = Tensor::from_data(
        TensorData::from([[[1.0f32, 2.0, 3.0], [4.0, 4.0, 4.0]]]),
        &device,
    );
    let y = ln.forward(x);
    // t0: mean=2, var=(1+0+1)/3=2/3; frac = 1/sqrt(2/3) = 1.2247448...
    //   [-1, 0, 1] * 1.2247448 = [-1.2247448, 0, 1.2247448]
    // t1: mean=4, var=0 → [0, 0, 0]
    let got = y.into_data().to_vec::<f32>().unwrap();
    let frac = (2.0f32 / 3.0).powf(-0.5);
    let exp = &[-frac, 0.0, frac, 0.0, 0.0, 0.0];
    for (g, e) in got.iter().zip(exp.iter()) {
        assert!((g - e).abs() < 1e-3, "got {g}, expected {e}");
    }
}
