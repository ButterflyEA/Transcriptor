//! Type-level and runtime checks for the backend alias module (T17).
//!
//! The `backends` module must provide:
//!   - `CpuWhisper`  = `Whisper<NdArray<f32>>`            (feature `ndarray`)
//!   - `WgpuWhisper` = `Whisper<Wgpu>`                    (feature `wgpu`)
//!   - `cpu_device()` device helper (ndarray CPU)
//!   - a runtime fallback helper that attempts a wgpu device without
//!     erroring when no GPU/adapter is available.

use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use whisper_burn::backends::{BackendChoice, default_device};
use whisper_burn::backends::{CpuWhisper, cpu_device};
use whisper_burn::model::whisper::Whisper;

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};

/// Compile-time assertion: `CpuWhisper` is *exactly* `Whisper<NdArray<f32>>`.
fn _assert_cpu_alias_type() {
    let _: Option<Whisper<NdArray<f32>>> = None::<CpuWhisper>;
}

/// Compile-time assertion: `WgpuWhisper` is *exactly* `Whisper<Wgpu>`.
#[cfg(feature = "wgpu")]
fn _assert_wgpu_alias_type() {
    let _: Option<Whisper<Wgpu>> = None::<whisper_burn::backends::WgpuWhisper>;
}

/// `cpu_device()` yields a device the ndarray CPU backend accepts.
#[test]
fn cpu_device_matches_ndarray_backend() {
    let device = cpu_device();
    let _: Tensor<NdArray<f32>, 1> = Tensor::ones([3], &device);
}

/// Without breaking, we can ask for a default device and always end up with
/// a usable device: wgpu when an adapter is available, ndarray CPU otherwise.
#[test]
fn default_device_never_errors() {
    let choice = default_device();
    match choice {
        BackendChoice::Cpu(device) => {
            let _: Tensor<NdArray<f32>, 1> = Tensor::ones([3], &device);
        }
        #[cfg(feature = "wgpu")]
        BackendChoice::Wgpu(device) => {
            let device: WgpuDevice = device;
            let _: Tensor<Wgpu, 1> = Tensor::ones([3], &device);
        }
        #[cfg(not(feature = "wgpu"))]
        _ => {}
    }
}
