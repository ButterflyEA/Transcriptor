//! Backend type aliases and runtime device fallback (T17).
//!
//! [`crate::model::whisper::Whisper`] is generic over a burn
//! [`Backend`](burn::tensor::backend::Backend). This module fixes the backend
//! for each supported runtime and exposes a helper that picks a usable device
//! at runtime, falling back to the CPU when no GPU adapter is available
//! instead of erroring.

use crate::model::whisper::Whisper;

/// Whisper specialized to the ndarray (pure CPU) backend.
#[cfg(feature = "ndarray")]
pub type CpuWhisper = Whisper<burn::backend::ndarray::NdArray<f32>>;

/// Whisper specialized to the wgpu (GPU / automatic adapter) backend.
#[cfg(feature = "wgpu")]
pub type WgpuWhisper = Whisper<burn::backend::wgpu::Wgpu>;

/// A backend device selected at runtime.
#[derive(Clone, Debug)]
pub enum BackendChoice {
    /// ndarray CPU device.
    #[cfg(feature = "ndarray")]
    Cpu(burn::backend::ndarray::NdArrayDevice),
    /// wgpu device (automatic adapter, incl. software/CPU adapters).
    #[cfg(feature = "wgpu")]
    Wgpu(burn::backend::wgpu::WgpuDevice),
}

/// Returns the ndarray CPU device.
#[cfg(feature = "ndarray")]
pub fn cpu_device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::default()
}

/// Attempts to initialise a wgpu device with an automatic adapter
/// (including software / CPU adapters). Returns `None` when burn cannot create
/// a device for the adapter (no GPU), so callers can fall back to a CPU device
/// instead of erroring.
#[cfg(feature = "wgpu")]
pub fn try_wgpu_device() -> Option<burn::backend::wgpu::WgpuDevice> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let usable = catch_unwind(AssertUnwindSafe(|| {
        let _: burn::tensor::Tensor<burn::backend::wgpu::Wgpu, 1> =
            burn::tensor::Tensor::ones([1], &device);
    }))
    .is_ok();
    usable.then_some(device)
}

/// Picks a usable device at runtime: prefers a wgpu device (automatic adapter,
/// including software / CPU adapters) when available, otherwise falls back to
/// the ndarray CPU device. Never errors — if no GPU adapter is available the
/// result is the CPU choice, so the caller can always proceed.
pub fn default_device() -> BackendChoice {
    #[cfg(feature = "wgpu")]
    if let Some(device) = try_wgpu_device() {
        return BackendChoice::Wgpu(device);
    }

    #[cfg(feature = "ndarray")]
    {
        BackendChoice::Cpu(cpu_device())
    }
    #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
    {
        panic!("no backend feature enabled")
    }
}
