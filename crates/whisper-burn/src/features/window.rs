pub const N_SAMPLES: usize = 480_000; // 30 s at 16 kHz
pub const N_FRAMES: usize = 3_000; // 30 s in 10 ms frames

pub fn split_chunks(samples: &[f32]) -> Vec<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + N_SAMPLES).min(samples.len());
        let mut chunk = vec![0.0f32; N_SAMPLES];
        chunk[..end - start].copy_from_slice(&samples[start..end]);
        out.push((chunk, end - start));
        start = end;
    }
    out
}

/// Window length granularity: 20 ms at 16 kHz, so the mel frame count is even
/// and the stride-2 conv in the encoder keeps every frame.
pub const WINDOW_QUANTUM: usize = 320;

/// Splits `len` samples into consecutive `(start, end)` windows of `window`
/// samples. The last window may be shorter and is not padded.
pub fn split_windows(len: usize, window: usize) -> Vec<(usize, usize)> {
    assert!(window > 0, "window must be positive");
    (0..len)
        .step_by(window)
        .map(|start| (start, (start + window).min(len)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_windows_covers_audio_without_padding() {
        assert_eq!(
            split_windows(48_000, 16_000),
            vec![(0, 16_000), (16_000, 32_000), (32_000, 48_000)]
        );
        assert_eq!(
            split_windows(40_000, 16_000),
            vec![(0, 16_000), (16_000, 32_000), (32_000, 40_000)]
        );
        assert!(split_windows(0, 16_000).is_empty());
    }
}
