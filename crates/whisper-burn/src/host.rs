//! Host memory capacity and which model will plausibly fit in it.
//!
//! This is advisory only. Nothing here blocks a download or refuses a run: the
//! footprints in [`ModelSize::approx_resident_bytes`] are a floor rather than a
//! total, and swap, shared memory and container limits all make real-world
//! capacity unknowable from inside the process. The purpose is to warn *before*
//! the kernel OOM-kills the app, which it does silently - no panic, no error,
//! every window simply closes.

use crate::config::ModelSize;

/// Fraction of available memory a model's estimated footprint may occupy.
///
/// The footprint excludes activations, the mel spectrogram and the KV cache, so
/// this reserve stands in for them. Transcribing with the page cache already
/// committed is technically survivable and practically useless.
const HEADROOM_NUM: u64 = 7;
const HEADROOM_DEN: u64 = 10;

/// What the machine reports about its own memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapacity {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// Whether a model's estimated footprint is within budget for this host.
pub fn fits(estimate_bytes: u64, available_bytes: u64) -> bool {
    estimate_bytes
        .checked_mul(HEADROOM_DEN)
        .is_some_and(|scaled| scaled <= available_bytes.saturating_mul(HEADROOM_NUM))
}

/// The largest model whose footprint fits, or `None` when even `tiny` does not.
pub fn recommend(capacity: &HostCapacity, sizes: &[ModelSize]) -> Option<ModelSize> {
    recommend_with(capacity, sizes, ModelSize::approx_resident_bytes)
}

/// As [`recommend`], but with a caller-supplied footprint source.
///
/// Callers holding a downloaded checkpoint should pass [`resident_bytes`] so
/// the recommendation is based on the real file. Mixing the two sources makes
/// the recommendation contradict the per-model verdicts shown next to it.
pub fn recommend_with<F>(capacity: &HostCapacity, sizes: &[ModelSize], estimate: F) -> Option<ModelSize>
where
    F: Fn(ModelSize) -> u64,
{
    sizes
        .iter()
        .copied()
        .filter(|m| fits(estimate(*m), capacity.available_bytes))
        .max_by_key(|m| estimate(*m))
}

/// The model's estimated footprint, preferring the real file on disk.
///
/// Once a checkpoint is downloaded its true size is known, so there is no
/// reason to estimate. A zero-length file is ignored: that is a failed or
/// still-running download, not a model that fits in no memory at all.
pub fn resident_bytes(size: ModelSize) -> u64 {
    #[cfg(feature = "weights")]
    if let Ok(dir) = crate::download::cache_dir(size)
        && let Ok(meta) = std::fs::metadata(dir.join("model.safetensors"))
        && meta.len() > 0
    {
        return meta.len();
    }
    size.approx_resident_bytes()
}

/// Reads the host's memory. `None` when detection fails or reports nonsense.
///
/// A zero `total_bytes` is treated as failure rather than as a real reading:
/// it would otherwise recommend `tiny` and imply the machine is broken, which
/// is a worse lie than staying quiet.
pub fn detect() -> Option<HostCapacity> {
    use sysinfo::System;

    let mut sys = System::new();
    // The first refresh can report zeroed counters on some platforms, so read
    // once to prime and again for the real values.
    sys.refresh_memory();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    if total_bytes == 0 {
        return None;
    }
    Some(HostCapacity { total_bytes, available_bytes })
}
