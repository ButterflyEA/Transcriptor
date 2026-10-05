//! Energy gate that keeps silent windows away from the model.

pub const DEFAULT_RMS: f32 = 0.01;
const FRAME_MS: u32 = 30;

/// True when at least two consecutive 30 ms frames exceed the threshold, so a
/// single click does not count as speech.
pub fn is_speech(pcm: &[f32], rate: u32, threshold_rms: f32) -> bool {
    let frame = (rate * FRAME_MS / 1000) as usize;
    if frame == 0 {
        return false;
    }
    let mut run = 0;
    for chunk in pcm.chunks_exact(frame) {
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt();
        if rms > threshold_rms {
            run += 1;
            if run >= 2 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}