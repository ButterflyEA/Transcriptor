use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::N_FRAMES;

const GOLDEN: &str = include_str!("fixtures/golden_tone.json");

fn tone_padded_30s() -> Vec<f32> {
    let mut a = vec![0.0f32; 480_000];
    for (i, s) in a.iter_mut().take(16_000).enumerate() {
        *s = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
    }
    a
}

#[test]
fn log_mel_matches_reference_tone() {
    let fe = FeatureExtractor::new(80, 16_000).unwrap();
    let m = fe.log_mel(&tone_padded_30s()).unwrap();

    let j: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(
        m.len(),
        80 * N_FRAMES,
        "mel must drop the trailing STFT frame like torch `stft[:, :-1]`"
    );
    for (c, col) in j["mel_cols"].as_object().unwrap() {
        let c: usize = c.parse().unwrap();
        let golden: Vec<f32> = col
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for (mm, v) in golden.iter().enumerate() {
            let got = m[mm * N_FRAMES + c];
            // naive O(n^2) DFT sums in a different order than torch's FFT, so
            // individual log-mel cells wobble a few e-3 (up to ~6e-3 on steep
            // spectral gradients); 1e-2 absorbs that while still pinning scale
            // and filter orientation tightly.
            assert!(
                (got - v).abs() < 1e-2,
                "col {c} mel-row {mm}: got {got} golden {v}"
            );
        }
    }
}
