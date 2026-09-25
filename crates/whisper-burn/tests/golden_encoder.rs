#![cfg(all(feature = "ndarray", feature = "audio", feature = "weights"))]

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::config::ModelSize;
use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::model::whisper::Whisper;

const GOLDEN: &str = include_str!("fixtures/golden_tone.json");

fn tone_padded_30s() -> Vec<f32> {
    let mut a = vec![0.0f32; 480_000];
    for (i, s) in a.iter_mut().take(16_000).enumerate() {
        *s = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
    }
    a
}

#[test]
#[ignore = "network (downloads tiny weights) + golden files"]
fn golden_encoder_matches_reference() {
    let device = NdArrayDevice::default();
    let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, device).unwrap();
    let fe = FeatureExtractor::new(w.dims.n_mels, 16_000).unwrap();
    let mel = fe.log_mel(&tone_padded_30s()).unwrap();
    assert_eq!(mel.len(), w.dims.n_mels * 3000);
    let mel: Tensor<NdArray<f32>, 3> =
        Tensor::from_data(TensorData::new(mel, [1, w.dims.n_mels, 3000]), &device);
    let enc = w.forward_encoder(mel);
    assert_eq!(enc.shape().dims(), [1, 1500, 384]);

    let flat = enc.into_data().to_vec::<f32>().unwrap();
    let j: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    for (r, rows) in j["enc_head"].as_object().unwrap() {
        let r: usize = r.parse().unwrap();
        let golden: Vec<f32> = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for (k, v) in golden.iter().enumerate() {
            let got = flat[r * 384 + k];
            assert!(
                (got - v).abs() < 3e-3,
                "enc row {r} k {k}: got {got} golden {v}"
            );
        }
    }
}
