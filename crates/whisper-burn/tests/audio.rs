#![cfg(feature = "audio")]
use std::io::Write as _;
use whisper_burn::audio::decode::decode_to_mono_f32;

fn write_wav(path: &std::path::Path, samples: &[f32], sr: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    // minimal PCM16 WAV header
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut hdr = Vec::new();
    hdr.extend(b"RIFF");
    hdr.extend((36 + data_len).to_le_bytes());
    hdr.extend(b"WAVE");
    hdr.extend(b"fmt ");
    hdr.extend(16u32.to_le_bytes());
    hdr.extend(1u16.to_le_bytes()); // PCM
    hdr.extend(1u16.to_le_bytes()); // mono
    hdr.extend(sr.to_le_bytes());
    hdr.extend((sr * 2).to_le_bytes());
    hdr.extend(2u16.to_le_bytes()); // block align
    hdr.extend(16u16.to_le_bytes()); // bits
    hdr.extend(b"data");
    hdr.extend(data_len.to_le_bytes());
    f.write_all(&hdr).unwrap();
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

#[test]
fn decodes_wav_mono() {
    let dir = std::env::temp_dir();
    let path = dir.join("whisper_test_tone.wav");
    let sr = 16000u32;
    let samples: Vec<f32> = (0..1600)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();
    write_wav(&path, &samples, sr);
    let (out, got_sr) = decode_to_mono_f32(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(got_sr, sr);
    assert_eq!(out.len(), samples.len());
    for (a, b) in out.iter().zip(samples.iter()) {
        assert!((a - b).abs() < 0.02);
    }
}

#[test]
fn decodes_mp3() {
    // mp3 needs a non-default symphonia feature (plan §T9: "any symphonia-
    // supported container/codec"); WAV passes but this one used to fail with
    // "probe: no suitable format reader found"
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_411000_s.mp3");
    let (out, got_sr) = decode_to_mono_f32(path).unwrap();
    assert!(got_sr >= 16_000, "expected a real sample rate, got {got_sr}");
    assert!(
        out.len() > got_sr as usize * 4,
        "expected several seconds of audio, got {} samples",
        out.len()
    );
}
