#[derive(Clone, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
        }
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn dft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_freq = n / 2 + 1;
    let pi2 = 2.0 * std::f32::consts::PI;
    (0..n_freq)
        .map(|k| {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (m, &x) in frame.iter().enumerate() {
                let ang = -pi2 * k as f32 * m as f32 / n as f32;
                re += x * ang.cos();
                im += x * ang.sin();
            }
            re * re + im * im
        })
        .collect()
}

pub fn stft_power(samples: &[f32], config: &StftConfig) -> Vec<Vec<f32>> {
    let pad = config.n_fft / 2;
    // torch.stft(center=True, pad_mode="reflect"): mirror `pad` samples each
    // side without repeating the edge (numpy `reflect`).
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for k in 0..pad {
        padded.push(samples[pad - k]);
    }
    padded.extend_from_slice(samples);
    for k in 0..pad {
        padded.push(samples[samples.len() - 2 - k]);
    }

    let n_frames = samples.len() / config.hop_length + 1;
    let window = hann(config.n_fft);

    let n_freq = config.n_fft / 2 + 1;
    let frame_power = |t: usize| -> Vec<f32> {
        let start = t * config.hop_length;
        let frame: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(b, &w)| padded[start + b] * w)
            .collect();
        // An all-zero frame (the padding of a short window) has exactly zero
        // power, so the DFT can be skipped without changing any value.
        if frame.iter().all(|&x| x == 0.0) {
            vec![0.0; n_freq]
        } else {
            dft_power(&frame)
        }
    };

    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(n_frames.max(1));
    let per = n_frames.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_frames)
            .step_by(per)
            .map(|lo| {
                let hi = (lo + per).min(n_frames);
                let frame_power = &frame_power;
                scope.spawn(move || (lo..hi).map(frame_power).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("stft worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hann(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect()
    }

    #[test]
    fn golden_small_frame_power() {
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let cfg = StftConfig {
            n_fft: 4,
            hop_length: 2,
        };
        let power = stft_power(&x, &cfg);
        assert_eq!(power.len(), 3);
        assert_eq!(power[0].len(), 3); // n_fft/2+1
        let w = hann(4); // [0.0, 0.5, 1.0, 0.5]
        assert_eq!(w, vec![0.0, 0.5, 1.0, 0.5]);
        // reflect padding (mirror without edge repeat): padded = [1,1, 1,1,1,1, 1,1]
        // every 4-window is [1,1,1,1], x*w = [0,0.5,1,0.5]
        //  X0 = 2.0, |X1|^2 = 1.0, |X2|^2 = 0.0
        for p in &power {
            assert!((p[0] - 4.0).abs() < 1e-4);
            assert!((p[1] - 1.0).abs() < 1e-4);
            assert!((p[2] - 0.0).abs() < 1e-4);
        }
    }

    #[test]
    fn sine_tone_peaks_at_bin() {
        let sr = 16000.0;
        let f = 400.0; // 400 Hz -> bin k = f * n_fft / sr = 10
        let n = 4000usize;
        let cfg = StftConfig {
            n_fft: 400,
            hop_length: 160,
        };
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin())
            .collect();
        let power = stft_power(&x, &cfg);
        let mid = power.len() / 2;
        let peak = power[mid]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 10);
        assert_eq!(power.len(), 26);
    }
}
