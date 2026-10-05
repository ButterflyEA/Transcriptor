use dictation_service::vad::*;

const RATE: u32 = 16_000;

fn tone(ms: u32, amp: f32) -> Vec<f32> {
    (0..RATE * ms / 1000)
        .map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE as f32).sin())
        .collect()
}

#[test]
fn zeros_and_low_noise_are_not_speech() {
    assert!(!is_speech(&vec![0.0; 16_000], RATE, DEFAULT_RMS));
    assert!(!is_speech(&tone(1000, 0.005), RATE, DEFAULT_RMS));
}

#[test]
fn a_tone_is_speech() {
    assert!(is_speech(&tone(500, 0.1), RATE, DEFAULT_RMS));
}

#[test]
fn a_short_click_is_not_speech() {
    let mut pcm = vec![0.0; 16_000];
    pcm[8_000..8_000 + 160].copy_from_slice(&tone(10, 0.5));
    assert!(!is_speech(&pcm, RATE, DEFAULT_RMS));
}