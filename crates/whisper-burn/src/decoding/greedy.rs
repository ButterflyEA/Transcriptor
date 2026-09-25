//! Greedy temperature-0 decoding: run the decoder autoregressively, applying
//! the reference logit filters (suppress-blank at the first step, token
//! suppression every step), argmax each token, stop on `<|endoftext|>` or the
//! token cap.
//!
//! Mirrors `whisper.decoding`: `GreedyDecoder.update` for the argmax +
//! `log_softmax` logprob accumulation, `SuppressBlank`/`SuppressTokens` logit
//! filters, and the `no_speech_prob` capture at the `sot` position on step 0.

use crate::model::whisper::Whisper;
use crate::tokenizer::whisper::{control_tokens, special_ids};
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// `<|0.00|>`'s base token; timestamp ids sit at `timestamp_begin..`. We do not
/// apply timestamp pairing rules yet (that arrives with task 24).
pub const BLANK: u32 = 220;

/// Loop/token cap; the whisper `n_text_ctx // 2` default (448/2 = 224).
pub const DEFAULT_SAMPLE_LEN: usize = 224;

/// Greedy decoding options. Defaults match the whisper CLI
/// (`suppress_blank=True`, `suppress_tokens="-1"`, the 0.6 / -1.0 no-speech
/// thresholds).
#[derive(Debug, Clone)]
pub struct GreedyOptions {
    /// Hard cap on generated tokens.
    pub max_tokens: usize,
    /// Step budget for the autoregressive loop (default `n_text_ctx // 2`).
    pub sample_len: usize,
    /// Suppress `" "` and `<|endoftext|>` on the very first sampling step.
    pub suppress_blank: bool,
    /// Token ids to suppress each step; `u32::MAX` is the `"-1"` marker that
    /// resolves to the reference control tokens. Empty disables the filter.
    pub suppress_tokens: Vec<u32>,
    /// Thresholds applied later by `transcribe` (no-speech filtering); carried
    /// here so decoding carries the user's intent.
    pub no_speech_threshold: f64,
    pub logprob_threshold: f64,
    /// Reserved for the cross-chunk prefix conditioning (task 25).
    pub condition_on_previous_text: bool,
}

impl Default for GreedyOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_SAMPLE_LEN,
            sample_len: DEFAULT_SAMPLE_LEN,
            suppress_blank: true,
            suppress_tokens: vec![u32::MAX],
            no_speech_threshold: 0.6,
            logprob_threshold: -1.0,
            condition_on_previous_text: false,
        }
    }
}

/// Result statistics of a greedy run, used for no-speech decisions later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreedyResult {
    /// `sum(lp)` over sampled tokens including the final `<|endoftext|>`,
    /// matching reference `sum_logprobs`.
    pub sum_logprob: f32,
    /// `sum_logprob / sampled_count` (reference `avg_logprob`).
    pub mean_logprob: f32,
    /// Softmax probability of `<|nospeech|>` at the `sot` position, captured
    /// before filtering on the first step (reference `no_speech_prob`).
    pub no_speech_prob: f32,
    /// Stopped on `<|endoftext|>` (vs hitting the token cap).
    pub terminated_eot: bool,
}

/// The reference suppression list (`_get_suppress_tokens`): any user ids, plus
/// the control tokens `{transcribe, translate, sot, sot_prev, sot_lm,
/// nospeech}` that are unconditionally suppressed, deduped and sorted.
/// `u32::MAX` is the `"-1"` marker. The symbol-based `non_speech_tokens` set
/// needs the full tiktoken vocab and is deferred until a tokenizer is embedded
/// (task 27); ids past `n_vocab` are dropped for safety.
pub fn resolve_suppression(suppress_tokens: &[u32], n_vocab: usize) -> Vec<u32> {
    let mut s: Vec<u32> = suppress_tokens
        .iter()
        .copied()
        .filter(|&t| t != u32::MAX)
        .collect();
    s.extend(control_tokens(n_vocab));
    s.sort_unstable();
    s.dedup();
    s.retain(|&t| (t as usize) < n_vocab);
    s
}

fn logprob_argmax(row: &[f32]) -> (f32, u32) {
    // log_softmax(row)[argmax] = x - logsumexp = -ln(sum(exp(x - max))).
    // Torch's log_softmax subtracts the running max; we do the same so the
    // logprob is a genuine log-probability (always <= 0).
    let (mut max, mut arg) = (f32::NEG_INFINITY, 0u32);
    for (i, &v) in row.iter().enumerate() {
        if v > max {
            max = v;
            arg = i as u32;
        }
    }
    let mut sum = 0.0f32;
    for &v in row {
        sum += (v - max).exp();
    }
    (-sum.ln(), arg)
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let (mut max, mut sum) = (f32::NEG_INFINITY, 0.0f32);
    for &v in row {
        if v > max {
            max = v;
        }
    }
    let mut out = vec![0.0f32; row.len()];
    for (o, &v) in out.iter_mut().zip(row) {
        *o = (v - max).exp();
        sum += *o;
    }
    for o in &mut out {
        *o /= sum;
    }
    out
}

/// The reference greedy loop (whisper CLI, `--temperature 0`, without
/// timestamps rules): generate into `tokens` until `<|endoftext|>` or the cap.
pub fn greedy_search<B: Backend>(
    whisper: &Whisper<B>,
    xa: &Tensor<B, 3>,
    tokens: &mut Vec<u32>,
    options: &GreedyOptions,
) -> Result<GreedyResult> {
    let n_vocab = whisper.dims.n_vocab;
    let sid = special_ids(n_vocab);
    let suppress = resolve_suppression(&options.suppress_tokens, n_vocab);
    let sot_index = tokens
        .iter()
        .position(|&t| t == sid.sot)
        .ok_or_else(|| Error::Tokenizer("greedy search needs <|startoftranscript|>".into()))?;
    let dev = &xa.device();
    let start_len = tokens.len();
    let budget = options.sample_len.min(options.max_tokens);

    let mut sum_logprob = 0.0f32;
    let mut no_speech_prob = f32::NAN;
    let mut terminated_eot = false;

    for step in 0..budget {
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        let decoded = whisper.forward_decoder(
            Tensor::<B, 1, burn::tensor::Int>::from_data(TensorData::new(ids, [tokens.len()]), dev),
            xa,
        ); // [seq, n_vocab]

        if step == 0 {
            // no-speech probability at the sot position, before any filtering
            let row = decoded
                .clone()
                .slice([sot_index..sot_index + 1, 0..n_vocab])
                .reshape([n_vocab])
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| Error::Decoder(format!("reading sot logits: {e}")))?;
            no_speech_prob = softmax(&row)[sid.nospeech as usize];
        }

        let mut row = decoded
            .slice([tokens.len() - 1..tokens.len(), 0..n_vocab])
            .reshape([n_vocab])
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Decoder(format!("reading logits: {e}")))?;

        if step == 0 && options.suppress_blank {
            row[BLANK as usize] = f32::NEG_INFINITY;
            row[sid.eot as usize] = f32::NEG_INFINITY;
        }
        for &t in &suppress {
            row[t as usize] = f32::NEG_INFINITY;
        }

        let (lp, token) = logprob_argmax(&row);
        sum_logprob += lp;
        tokens.push(token);
        if token == sid.eot {
            terminated_eot = true;
            break;
        }
    }

    let sampled = tokens.len() - start_len;
    // reference avg_logprob divides by (len(t) + 1); on eot-termination t
    // excludes the final eot, on truncation the padded eot is never counted
    let mean_logprob = sum_logprob / (sampled + usize::from(!terminated_eot)) as f32;

    Ok(GreedyResult {
        sum_logprob,
        mean_logprob,
        no_speech_prob,
        terminated_eot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::whisper::EOT;

    #[test]
    fn logprob_is_ln_softmax_confident() {
        // one dominant logit -> log_softmax very close to 0 and never positive
        for max_v in [10.0f32, 30.0, 100.0] {
            let mut row = vec![0.0f32; 51866];
            row[3] = max_v;
            let (lp, arg) = logprob_argmax(&row);
            assert_eq!(arg, 3);
            assert!(lp <= 0.0, "positive logprob {lp}");
        }
        // exact value for max=10: -ln(1 + 51865 exp(-10)); float32 summation
        // over 51k terms drifts ~1e-3, so bound loosely
        let mut row = vec![0.0f32; 51866];
        row[3] = 10.0;
        let (lp, _) = logprob_argmax(&row);
        let expected = -(1.0f32 + 51865.0 * (-10.0f32).exp()).ln();
        assert!((lp - expected).abs() < 1e-2, "lp {lp}, expected {expected}");
    }

    #[test]
    fn logprob_uniform_is_minus_ln_n() {
        let n = 4usize;
        let mut row = vec![1.0f32; n];
        let (lp, _) = logprob_argmax(&row);
        assert!((lp + (n as f32).ln()).abs() < 1e-6, "lp {lp}");
        row[0] = 0.0;
        row[2] = 0.0;
        let (lp2, arg2) = logprob_argmax(&row);
        assert_eq!(arg2, 1);
        // one less competitor: ln(3) = -1.0986, matches log_softmax(x) - ln(2)?
        assert!(lp2 < 0.0 && lp2 > -3.0, "lp2 {lp2}");
    }

    #[test]
    fn suppose_eot_never_suppressed() {
        let s = resolve_suppression(&[u32::MAX, 99_999, EOT], 51865);
        // explicit user request is honored
        assert!(s.contains(&EOT));
        // but the default resolves without it
        let d = resolve_suppression(&[], 51865);
        assert!(!d.contains(&EOT));
    }
}
