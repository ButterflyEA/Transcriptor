use burn::module::{Module, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// A&S 7.1.26 erf approximation (max abs error ~1.5e-7 in f64; <1e-6 in f32).
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn erf_tensor<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x_abs = x.clone().abs();
    let t = x_abs.clone().mul_scalar(0.3275911).add_scalar(1.0).recip();
    let x2 = x_abs.clone().mul(x_abs);
    let y = t
        .clone()
        .mul_scalar(1.061405429)
        .sub_scalar(1.453152027)
        .mul(t.clone())
        .add_scalar(1.421413741)
        .mul(t.clone())
        .sub_scalar(0.284496736)
        .mul(t.clone())
        .add_scalar(0.254829592)
        .mul(t)
        .mul(x2.neg().exp())
        .neg()
        .add_scalar(1.0);
    y.mul(x.sign())
}

/// Exact GELU via erf: `0.5 * x * (1 + erf(x / sqrt(2)))`.
pub fn gelu_erf<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let erf = erf_tensor(x.clone().div_scalar(std::f64::consts::SQRT_2));
    x.mul(erf.add_scalar(1.0)).mul_scalar(0.5)
}

#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> Linear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = x.matmul(self.weight.val().transpose().unsqueeze::<3>());
        if let Some(b) = &self.bias {
            y = y.add(b.val().unsqueeze::<3>());
        }
        y
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    /// Epsilon added to the variance before normalizing (reference uses 1e-5).
    pub epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(weight: Param<Tensor<B, 1>>, bias: Param<Tensor<B, 1>>) -> Self {
        Self {
            weight,
            bias,
            epsilon: 1e-5,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let xm = x.clone().sub(mean.clone());
        let var = xm.clone().mul(xm.clone()).mean_dim(2);
        let denom = var.add_scalar(self.epsilon).powf_scalar(-0.5);
        let norm = xm.mul(denom);
        norm.mul(self.weight.val().unsqueeze::<3>())
            .add(self.bias.val().unsqueeze::<3>())
    }
}
