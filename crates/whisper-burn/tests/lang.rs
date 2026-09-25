#![cfg(all(feature = "ndarray", feature = "weights"))]

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::{detect_language, language_window, num_languages};
use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::split_chunks;
use whisper_burn::model::whisper::Whisper;

type B = NdArray<f32>;

#[test]
fn language_window_matches_reference() {
    // tiny/base/...: 99 standard codes at 50259..50357
    let w = language_window(51865);
    assert_eq!(w.len(), 99);
    assert_eq!(w[0], (50259, "en"));
    assert_eq!(w[98], (50357, "su"));
    assert_eq!(num_languages(51865), 99);

    // large-v3 / turbo: + <|yue|> as the 100th language token at 50358
    let w3 = language_window(51866);
    assert_eq!(w3.len(), 100);
    assert_eq!(w3[0], (50259, "en"));
    assert_eq!(w3[99], (50358, "yue"));
    assert_eq!(num_languages(51866), 100);
}

#[test]
#[ignore = "network"]
fn detects_english_silence() {
    let dev = NdArrayDevice::default();
    let w = Whisper::<B>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    let pcm = vec![0.0f32; 480_000];
    let f = FeatureExtractor::new(80, 16000).unwrap();
    let chunks = split_chunks(&pcm);
    let mel = f.log_mel(&chunks[0].0).unwrap();
    let xa = w.forward_encoder(Tensor::from_data(TensorData::new(mel, [1, 80, 3000]), &dev));
    let info = detect_language(&w, &xa).unwrap();
    assert_eq!(info.language, "en");
    assert_eq!(info.token, 50259);
    // comes from real decoder logits, not a "always en" placeholder
    assert!(info.prob > 0.0 && info.prob < 1.0, "prob {}", info.prob);
}
