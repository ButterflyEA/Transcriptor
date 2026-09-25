use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::{N_FRAMES, N_SAMPLES, split_chunks};

const FILTERS_80: &[u8] = include_bytes!("../assets/mel_filters_80.bin");
const FILTERS_128: &[u8] = include_bytes!("../assets/mel_filters_128.bin");

#[test]
fn assets_have_expected_shape() {
    assert_eq!(FILTERS_80.len(), 80 * 201 * 4);
    assert_eq!(FILTERS_128.len(), 128 * 201 * 4);
}

#[test]
fn chunking_pads_to_30s() {
    let s = vec![0.0f32; N_SAMPLES + 1000];
    let chunks = split_chunks(&s);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].0.len(), N_SAMPLES);
    assert_eq!(chunks[0].1, N_SAMPLES);
    assert_eq!(chunks[1].0.len(), N_SAMPLES);
    assert_eq!(chunks[1].1, 1000);
    assert!(chunks[1].0[N_SAMPLES - 1] == 0.0);
    let _ = N_FRAMES;
}

#[test]
fn log_mel_shape_and_math() {
    let fe = FeatureExtractor::new(80, 16000).unwrap();
    let s = vec![0.0f32; 16000];
    let m = fe.log_mel(&s).unwrap();
    // torch stft (center=True) gives len/hop + 1 frames; whisper drops the last.
    let n_frames = 16000 / 160;
    assert_eq!(m.len(), 80 * n_frames);
    assert!(m.iter().all(|&v| v > -30.0 && v < 30.0));
    let dc = vec![1.0f32; 16000];
    let m2 = fe.log_mel(&dc).unwrap();
    assert!(m2.iter().all(|&v| v.is_finite()));
}

#[test]
fn extract_features_helper_matches_default_extractor() {
    let tone: Vec<f32> = (0..16_000)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
        .collect();
    let fe = FeatureExtractor::new(80, 16_000).unwrap();
    let expected = fe.log_mel(&tone).unwrap();
    let got = whisper_burn::features::extract_features(&tone).unwrap();
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-6, "got {a} expected {b}");
    }
}
