use super::stft::{StftConfig, stft_power};
use crate::{Error, Result};

pub struct FeatureExtractor {
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    filters: Vec<f32>,
    sample_rate: u32,
}

fn load_filters(n_mels: usize) -> Result<Vec<f32>> {
    let bytes = match n_mels {
        80 => include_bytes!("../../assets/mel_filters_80.bin").as_slice(),
        128 => include_bytes!("../../assets/mel_filters_128.bin").as_slice(),
        other => return Err(Error::UnsupportedFormat(format!("n_mels={other}"))),
    };
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

impl FeatureExtractor {
    pub fn new(n_mels: usize, sample_rate: u32) -> Result<Self> {
        let filters = load_filters(n_mels)?;
        Ok(Self {
            n_mels,
            n_fft: 400,
            hop_length: 160,
            filters,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn log_mel(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let cfg = StftConfig {
            n_fft: self.n_fft,
            hop_length: self.hop_length,
        };
        let power = stft_power(samples, &cfg);
        // torch stft produces len/hop + 1 frames (center=True); whisper drops
        // the trailing frame (`stft[:, :-1]`), so we keep exactly len/hop.
        let n_frames = samples.len() / self.hop_length;
        let n_freq = self.n_fft / 2 + 1;
        let mut mel = vec![0.0f32; self.n_mels * n_frames];
        for m in 0..self.n_mels {
            let fb = &self.filters[m * n_freq..(m + 1) * n_freq];
            for t in 0..n_frames {
                let mut acc = 0.0f32;
                for (b, &p) in power[t].iter().enumerate() {
                    acc += fb[b] * p;
                }
                mel[m * n_frames + t] = acc;
            }
        }
        // log10 with floor, then max-8 clamp, then (x+4)/4
        for v in mel.iter_mut() {
            let x = (*v).max(1e-10).log10();
            *v = x;
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = max - 8.0;
        for v in mel.iter_mut() {
            *v = ((*v).max(floor) + 4.0) / 4.0;
        }
        Ok(mel)
    }
}

/// Default 80-mel / 16 kHz feature extraction over raw PCM, matching whisper's
/// `log_mel_spectrogram` shape `[n_mels, frames]`.
pub fn extract_features(audio_pcm: &[f32]) -> Result<Vec<f32>> {
    FeatureExtractor::new(80, 16_000)?.log_mel(audio_pcm)
}
