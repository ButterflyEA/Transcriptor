#![cfg(all(feature = "ndarray", feature = "weights"))]

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::beam::{BeamOptions, beam_search};
use whisper_burn::decoding::greedy::{GreedyOptions, greedy_search};
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

fn trim_eot(mut tokens: Vec<u32>) -> Vec<u32> {
    if tokens.last() == Some(&EOT) {
        tokens.pop();
    }
    tokens
}

#[test]
#[ignore = "network"]
fn beam_one_matches_greedy_on_silence() {
    let (w, xa) = silence_features();

    let mut greedy_tokens = vec![SOT, TRANSCRIBE];
    let greedy_opts = GreedyOptions {
        max_tokens: 8,
        sample_len: 8,
        ..Default::default()
    };
    let greedy_res = greedy_search(&w, &xa, &mut greedy_tokens, &greedy_opts).unwrap();

    let mut beam_tokens = vec![SOT, TRANSCRIBE];
    let beam_opts = BeamOptions {
        beam_size: 1,
        max_tokens: 8,
        ..Default::default()
    };
    let beam_res = beam_search(&w, &xa, &mut beam_tokens, &beam_opts).unwrap();

    // beam 1 is bit-identical to greedy modulo the trailing eot convention
    assert_eq!(
        trim_eot(beam_tokens.clone()),
        trim_eot(greedy_tokens.clone())
    );
    // prefix + 8 generated + finalize's padded eot
    assert!(beam_tokens.len() <= 11, "over budget: {beam_tokens:?}");

    // best sequence starts with the given sot_sequence and ends on eot
    assert_eq!(&beam_tokens[..2], &[SOT, TRANSCRIBE]);
    assert_eq!(
        *beam_tokens.last().unwrap(),
        EOT,
        "beam must finalize with eot"
    );

    // stats are genuine log-probabilities; the two paths agree numerically
    assert!((beam_res.sum_logprob - greedy_res.sum_logprob).abs() < 1e-3);
    assert!((beam_res.mean_logprob - greedy_res.mean_logprob).abs() < 1e-3);
    assert!(
        (0.0..=1.0).contains(&beam_res.no_speech_prob),
        "{}",
        beam_res.no_speech_prob
    );
}

#[test]
#[ignore = "network"]
fn beam_two_keeps_k_hypotheses_on_silence() {
    let (w, xa) = silence_features();

    let mut tokens = vec![SOT, TRANSCRIBE];
    let opts = BeamOptions {
        beam_size: 2,
        max_tokens: 6,
        ..Default::default()
    };
    let res = beam_search(&w, &xa, &mut tokens, &opts).unwrap();

    assert!(tokens.len() > 2, "no tokens sampled: {tokens:?}");
    assert!(tokens.len() <= 9, "over budget: {tokens:?}");
    assert_eq!(&tokens[..2], &[SOT, TRANSCRIBE]);
    assert_eq!(*tokens.last().unwrap(), EOT, "beam must finalize with eot");

    // only a single trailing eot; no control tokens slipped through
    let sampled = &tokens[2..tokens.len().saturating_sub(1)];
    let forbidden = [TRANSLATE, TRANSCRIBE, SOT, STARTOPREV, STARTOFLM, NOSPEECH];
    assert!(
        sampled.iter().all(|&t| !forbidden.contains(&t)),
        "control token leaked: {tokens:?}"
    );
    assert!(!sampled.contains(&EOT), "eot mid-stream: {tokens:?}");

    assert!(res.sum_logprob <= 0.0 && res.sum_logprob.is_finite());
    assert!(res.mean_logprob <= 0.0 && res.mean_logprob.is_finite());
    assert!(
        (0.0..=1.0).contains(&res.no_speech_prob),
        "{}",
        res.no_speech_prob
    );
}
