#![cfg(all(feature = "ndarray", feature = "weights"))]

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::greedy::{GreedyOptions, greedy_search, resolve_suppression};
use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::split_chunks;
use whisper_burn::model::whisper::Whisper;
use whisper_burn::tokenizer::whisper::{
    EOT, NOSPEECH, SOT, STARTOFLM, STARTOPREV, TRANSCRIBE, TRANSLATE,
};

type B = NdArray<f32>;

fn silence_features() -> (
    whisper_burn::model::whisper::Whisper<NdArray<f32>>,
    Tensor<NdArray<f32>, 3>,
) {
    let dev = NdArrayDevice::default();
    let w = Whisper::<B>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    let fe = FeatureExtractor::new(w.dims.n_mels, 16000).unwrap();
    let pcm = vec![0.0f32; 480_000];
    let chunks = split_chunks(&pcm);
    let mel = fe.log_mel(&chunks[0].0).unwrap();
    let xa = w.forward_encoder(Tensor::from_data(
        TensorData::new(mel, [1, w.dims.n_mels, 3000]),
        &dev,
    ));
    (w, xa)
}

#[test]
fn suppression_default_blocks_control_tokens() {
    // even an empty option list adds the reference control tokens, sorted+deduped
    let empty = resolve_suppression(&[], 51865);
    assert!(empty.contains(&SOT));
    assert!(empty.contains(&TRANSCRIBE));
    assert_eq!(
        empty.iter().filter(|&&t| empty.contains(&t)).count().min(1),
        1
    );
    let sorted = {
        let mut v = empty.clone();
        v.sort_unstable();
        v.dedup();
        v
    };
    assert_eq!(empty, sorted);

    // "-1" (u32::MAX) resolves to the same set
    assert_eq!(resolve_suppression(&[u32::MAX], 51865), empty);

    // out-of-vocab ids are dropped; eot is never suppressed
    let custom = resolve_suppression(&[u32::MAX, 999_999], 51865);
    assert!(custom.iter().all(|&t| t < 51865));
    assert!(!custom.contains(&EOT));
}

#[test]
#[ignore = "network"]
fn greedy_silence_decodes_real_tokens() {
    let (w, xa) = silence_features();

    let mut tokens = vec![SOT, TRANSCRIBE];
    let opts = GreedyOptions {
        max_tokens: 8,
        sample_len: 8,
        ..Default::default()
    };
    let res = greedy_search(&w, &xa, &mut tokens, &opts).unwrap();

    // the greedy loop sampled real tokens past the [sot, transcribe] prefix
    assert!(tokens.len() > 2, "no tokens sampled: {tokens:?}");
    assert!(tokens.len() <= 10, "over budget: {tokens:?}");

    // suppression never lets filtered control tokens back in
    let sampled = &tokens[2..];
    let forbidden = [TRANSLATE, TRANSCRIBE, SOT, STARTOPREV, STARTOFLM, NOSPEECH];
    assert!(
        sampled.iter().all(|&t| !forbidden.contains(&t)),
        "control token leaked: {tokens:?}"
    );
    // <|endoftext|> is only ever the final token
    if sampled.contains(&EOT) {
        assert_eq!(
            *tokens.last().unwrap(),
            EOT,
            "eot must end the run: {tokens:?}"
        );
    }

    // log-probability stats are real and no-speech prob is a probability
    assert!(res.sum_logprob <= 0.0 && res.sum_logprob.is_finite());
    assert!(res.mean_logprob <= 0.0 && res.mean_logprob.is_finite());
    assert!(
        (0.0..=1.0).contains(&res.no_speech_prob),
        "{}",
        res.no_speech_prob
    );
}
