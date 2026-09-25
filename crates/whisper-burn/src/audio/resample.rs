use crate::{Error, Result};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType};

pub fn resample_to_16k(samples: &[f32], in_sr: u32, out_sr: u32) -> Result<Vec<f32>> {
    if in_sr == out_sr {
        return Ok(samples.to_vec());
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, &params, 1024, 1, FixedAsync::Input)
        .map_err(|e| Error::Resample(e.to_string()))?;
    let input = InterleavedSlice::new(samples, 1, samples.len())
        .map_err(|e| Error::Resample(e.to_string()))?;
    let out = resampler
        .process_all(&input, samples.len(), None)
        .map_err(|e| Error::Resample(e.to_string()))?;
    Ok(out.take_data())
}

#[cfg(all(test, feature = "audio"))]
mod tests {
    use super::resample_to_16k;

    #[test]
    fn noop_for_16k() {
        let s = vec![0.0f32; 100];
        let out = resample_to_16k(&s, 16000, 16000).unwrap();
        assert_eq!(out, s);
    }

    #[test]
    fn upsamples_8k_to_16k_length() {
        let sr = 8000u32;
        let f = 1000.0;
        let s: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect();
        let out = resample_to_16k(&s, sr, 16000).unwrap();
        let ratio = out.len() as f32 / s.len() as f32;
        assert!(ratio > 1.9, "expected ~2x length, got {ratio}");
        assert!(ratio < 2.1, "expected ~2x length, got {ratio}");
    }

    #[test]
    fn preserves_sine_energy() {
        let sr = 8000u32;
        let f = 1000.0;
        let s: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect();
        let out = resample_to_16k(&s, sr, 16000).unwrap();
        let rms_in: f64 =
            (s.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / s.len() as f64).sqrt();
        let rms_out: f64 =
            (out.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / out.len() as f64).sqrt();
        assert!(
            (rms_out - rms_in).abs() / rms_in < 0.05,
            "energy drift: in {rms_in}, out {rms_out}"
        );
    }
}
