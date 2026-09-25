//! Language detection: run the decoder one step on the `<|startoftranscript|>`
//! token and read the logits restricted to the language-token window.
//!
//! Mirrors the reference `whisper.decoding.detect_language`: feed a single
//! `[sot]` token, mask every non-language logit to `-inf`, then softmax. With
//! all other logits `-inf` the softmax is identical to a softmax over the
//! language window alone, and the argmax is over the window either way.

use crate::model::whisper::Whisper;
use crate::tokenizer::whisper::{language_codes, special_ids};
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// `<|startoftranscript|>`.
pub const SOT: u32 = 50258;
/// `<|en|>`, the first language token.
pub const LANGUAGE_BASE: u32 = 50259;

/// True when `n_vocab` has language tokens, matching the reference
/// `Whisper.is_multilingual` (`n_vocab >= 51865`). All sizes we ship are.
pub fn is_multilingual(n_vocab: usize) -> bool {
    n_vocab >= 51865
}

/// Number of language tokens, matching the reference `Whisper.num_languages`
/// (`n_vocab - 51765 - 1`): 99 standard codes, plus `yue` for large-v3.
pub fn num_languages(n_vocab: usize) -> usize {
    if is_multilingual(n_vocab) {
        special_ids(n_vocab).n_languages
    } else {
        0
    }
}

/// The `(token, code)` pairs of the language-token window for a given vocab
/// size: `<|{code}|>` ids `50259..` in `LANGUAGES` order, plus `<|yue|>` at
/// 50358 for large-v3.
pub fn language_window(n_vocab: usize) -> Vec<(u32, &'static str)> {
    language_codes(num_languages(n_vocab))
        .into_iter()
        .enumerate()
        .map(|(i, code)| (LANGUAGE_BASE + i as u32, code))
        .collect()
}

/// The detected language: argmax over the window of the 1-step `[sot]` logits,
/// with `prob` the softmax of the window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanguageInfo {
    pub token: u32,
    pub language: &'static str,
    pub prob: f32,
}

/// Detects the spoken language from `xa`, the already-encoded audio features
/// (`[1, n_audio_ctx, n_audio_state]`).
pub fn detect_language<B: Backend>(
    whisper: &Whisper<B>,
    xa: &Tensor<B, 3>,
) -> Result<LanguageInfo> {
    let window = language_window(whisper.dims.n_vocab);
    if window.is_empty() {
        // English-only checkpoint: what `transcribe` would have defaulted to.
        return Ok(LanguageInfo {
            token: LANGUAGE_BASE,
            language: "en",
            prob: 1.0,
        });
    }

    let tokens = Tensor::<B, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![SOT as i64], [1]),
        &xa.device(),
    );
    let logits = whisper.forward_decoder(tokens, xa); // [1, n_vocab]
    let row = logits
        .reshape([whisper.dims.n_vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| Error::Decoder(format!("reading language logits: {e}")))?;

    let mut max = f32::NEG_INFINITY;
    for &(token, _) in &window {
        let v = row[token as usize];
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    let mut exps = vec![0.0f32; window.len()];
    for (e, &(token, _)) in exps.iter_mut().zip(&window) {
        *e = (row[token as usize] - max).exp();
        sum += *e;
    }
    let mut best = 0usize;
    let mut best_p = exps[0] / sum;
    for (i, e) in exps.iter().enumerate().skip(1) {
        let p = e / sum;
        if p > best_p {
            best = i;
            best_p = p;
        }
    }
    Ok(LanguageInfo {
        token: window[best].0,
        language: window[best].1,
        prob: best_p,
    })
}
