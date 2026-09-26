#![cfg(feature = "ndarray")]

#[allow(dead_code)]
mod common;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::greedy::{GreedyOptions, greedy_search, resolve_suppression};
use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::split_chunks;
use whisper_burn::model::whisper::Whisper;
use whisper_burn::tokenizer::whisper::{EOT, SOT, TIMESTAMP_BEGIN, TRANSCRIBE};
#[cfg(feature = "weights")]
use whisper_burn::tokenizer::whisper::{NOSPEECH, STARTOFLM, STARTOPREV, TRANSLATE};

type B = NdArray<f32>;

// Downloads a real checkpoint, so it needs the `weights` (ureq) feature on
// top of being `#[ignore]`d. The offline tests below load a fake checkpoint
// from disk and must stay runnable without it.
#[cfg(feature = "weights")]
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

fn fake_silence_features(w: &Whisper<B>, dev: &NdArrayDevice) -> Tensor<B, 3> {
    let fe = FeatureExtractor::new(w.dims.n_mels, 16000).unwrap();
    let pcm = vec![0.0f32; 480_000];
    let chunks = split_chunks(&pcm);
    let mel = fe.log_mel(&chunks[0].0).unwrap();
    w.forward_encoder(Tensor::from_data(
        TensorData::new(mel, [1, w.dims.n_mels, 3000]),
        dev,
    ))
}

/// The fake checkpoint is all zeros, so the decoder emits a flat logits row.
/// On a flat row the reference's fifth rule decides: the summed
/// log-probability of all ~1500 timestamp tokens beats any single text token,
/// so text is suppressed and a timestamp is forced. Unwired, argmax would
/// return the lowest surviving text id instead.
#[test]
fn timestamp_rules_are_applied_during_greedy_decoding() {
    let tmp = std::env::temp_dir().join(format!("wburn_greedy_ts_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    common::write_fake_checkpoint(ModelSize::Tiny, &tmp);
    let dev = NdArrayDevice::default();
    let w = Whisper::<B>::load(ModelSize::Tiny, &tmp, dev.clone()).unwrap();
    let xa = fake_silence_features(&w, &dev);

    let mut tokens = vec![SOT, TRANSCRIBE];
    let opts = GreedyOptions {
        max_tokens: 8,
        sample_len: 8,
        ..Default::default()
    };
    greedy_search(&w, &xa, &mut tokens, &opts).unwrap();
    let sampled = &tokens[2..];
    let _ = std::fs::remove_dir_all(&tmp);

    // rule 4: the first generated token must be a timestamp, capped at 1.0 s
    let first = sampled[0];
    assert!(
        (TIMESTAMP_BEGIN..TIMESTAMP_BEGIN + 51).contains(&first),
        "first generated token must be a timestamp <= 1.00s, got {first}"
    );

    // rule 2: a lone timestamp reads as a closed pair, so text does follow
    assert!(
        sampled.iter().any(|&t| t < TIMESTAMP_BEGIN),
        "text must not be permanently suppressed: {sampled:?}"
    );
    // Adjacent timestamps are legal and are how a pair closes - that is the
    // ts/text/ts/ts/text rhythm `assemble_segments` pairs positionally. What
    // rule 2 forbids is text in either illegal slot:
    //   * a closed pair (ts, ts) must be followed by text;
    //   * an open pair (text, ts) must be closed by a timestamp.
    for i in 2..sampled.len() {
        if sampled[i - 2] >= TIMESTAMP_BEGIN && sampled[i - 1] >= TIMESTAMP_BEGIN {
            assert!(
                sampled[i] < TIMESTAMP_BEGIN,
                "closed pair must be followed by text: {sampled:?}"
            );
        }
    }
    for i in 1..sampled.len().saturating_sub(1) {
        if sampled[i - 1] < TIMESTAMP_BEGIN && sampled[i] >= TIMESTAMP_BEGIN {
            assert!(
                sampled[i + 1] >= TIMESTAMP_BEGIN,
                "an open pair must be closed by a timestamp: {sampled:?}"
            );
        }
    }

    // rule 3: timestamps never decrease
    let stamps: Vec<u32> = sampled
        .iter()
        .copied()
        .filter(|&t| t >= TIMESTAMP_BEGIN)
        .collect();
    assert!(
        stamps.windows(2).all(|w| w[0] <= w[1]),
        "timestamps decreased: {stamps:?}"
    );
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
#[cfg(feature = "weights")]
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
