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
