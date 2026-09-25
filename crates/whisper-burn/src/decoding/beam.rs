//! Deterministic beam search decoding (temperature 0).
//!
//! Mirrors `whisper.decoding`: `BeamSearchDecoder` keeps `beam_size`
//! hypotheses, expands each by the top `beam_size + 1` next-token log-probs,
//! scores full sequences by cumulative `sum_logprob`, keeps the best non-eot
//! as active beams, and collects eot-completed sequences (capped by
//! `round(beam_size * patience)`). Finally the finished set is padded with the
//! best unfinished hypotheses (each + `<|endoftext|>`) and ranked by
//! `sum_logprob / sampled_len` (`MaximumLikelihoodRanker` with the default
//! length penalty), mirroring the reference output selection.
//!
//! Note: the reference `best_of` (random sampling at temperature > 0) is
//! carried for API parity but is unused here because decoding is greedy.

use std::cmp::Ordering;

use crate::model::whisper::Whisper;
use crate::tokenizer::whisper::special_ids;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use super::greedy::resolve_suppression;
use super::greedy::{BLANK, DEFAULT_SAMPLE_LEN};

/// Beam search decoding options. Defaults match the whisper CLI
/// (`beam_size=5`, `patience=1.0`, `suppress_blank=True`,
/// `suppress_tokens="-1"`).
#[derive(Debug, Clone)]
pub struct BeamOptions {
    /// Number of hypotheses kept alive at each step.
    pub beam_size: usize,
    /// Reference `best_of` (sampling at `temperature > 0`); kept for API
    /// parity, unused by the deterministic path.
    pub best_of: usize,
    /// Beam patience; `max_candidates = round(beam_size * patience)`. The
    /// run stops when that many sequences finish (patience 1.0 == stop as soon
    /// as `beam_size` sequences emit `<|endoftext|>`).
    pub patience: f32,
    /// Hard cap on generated tokens.
    pub max_tokens: usize,
    /// Suppress `" "` and `<|endoftext|>` on the very first sampling step.
    pub suppress_blank: bool,
    /// Token ids to suppress each step; `u32::MAX` is the `"-1"` marker that
    /// resolves to the reference control tokens. Empty disables the filter.
    pub suppress_tokens: Vec<u32>,
    /// Reserved for the cross-chunk prefix conditioning (task 25).
    pub condition_on_previous_text: bool,
}

impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            beam_size: 5,
            best_of: 5,
            patience: 1.0,
            max_tokens: DEFAULT_SAMPLE_LEN,
            suppress_blank: true,
            suppress_tokens: vec![u32::MAX],
            condition_on_previous_text: false,
        }
    }
}

/// Result statistics of a beam run, used for no-speech decisions later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamResult {
    /// `sum(lp)` over the winning sequence's sampled tokens including the
    /// final `<|endoftext|>`, matching reference `sum_logprobs`.
    pub sum_logprob: f32,
    /// `sum_logprob / (sampled_len + 1)` (reference `avg_logprob`).
    pub mean_logprob: f32,
    /// Softmax probability of `<|nospeech|>` at the `sot` position, captured
    /// before filtering on the first step (reference `no_speech_prob`).
    pub no_speech_prob: f32,
    /// The winning sequence ended on a sampled `<|endoftext|>` (vs the padded
    /// eot appended by `finalize`).
    pub terminated_eot: bool,
}

/// A running hypothesis: the full token sequence (caller prefix + generated
/// tokens) and the cumulative log-prob of the generated part.
#[derive(Debug, Clone)]
struct Hypothesis {
    tokens: Vec<u32>,
    sum_logprob: f32,
}

/// log_softmax over a logits row (torch semantics: subtract the running max).
fn log_softmax_row(row: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &v in row {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    let mut out = vec![0.0f32; row.len()];
    for (o, &v) in out.iter_mut().zip(row) {
        *o = (v - max).exp();
        sum += *o;
    }
    for o in &mut out {
        *o = o.ln() - sum.ln();
    }
    out
}

/// Top-`k` (logprob, token) pairs of a row, descending.
fn topk_logprobs(row: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut scored: Vec<(f32, u32)> = row
        .iter()
        .enumerate()
        .map(|(i, &lp)| (lp, i as u32))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored
}

/// A candidate sequence scored by cumulative log-prob.
type Scored = (Vec<u32>, f32);

/// STEP 1 of `BeamSearchDecoder.update`: for each beam take the top
/// `beam_size + 1` next-token log-probs and extend the sequence, scoring by
/// cumulative sum. Duplicate sequences behave like the reference dict: the
/// last assignment wins.
fn expand_candidates(
    beams: &[Hypothesis],
    step_logprobs: &[Vec<f32>],
    beam_size: usize,
) -> Vec<Scored> {
    let k = beam_size + 1;
    let mut scores: Vec<Scored> = Vec::new();
    for (beam, row) in beams.iter().zip(step_logprobs) {
        for (lp, token) in topk_logprobs(row, k) {
            let mut seq = beam.tokens.clone();
            seq.push(token);
            let score = beam.sum_logprob + lp;
            if let Some(entry) = scores.iter_mut().find(|(s, _)| *s == seq) {
                entry.1 = score;
            } else {
                scores.push((seq, score));
            }
        }
    }
    scores
}

/// STEP 2 of `BeamSearchDecoder.update`: rank the candidates by cumulative
/// score and keep the best `beam_size` non-eot as active; eot sequences are
/// collected as finished. Like the reference, iteration stops once `beam_size`
/// actives are saved, so only the eots ranked above that point are collected.
fn select_candidates(
    mut candidates: Vec<Scored>,
    beam_size: usize,
    eot: u32,
) -> (Vec<Scored>, Vec<Scored>) {
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut active = Vec::with_capacity(beam_size);
    let mut finished = Vec::new();
    let mut saved = 0usize;
    for (seq, score) in candidates {
        if seq.last() == Some(&eot) {
            finished.push((seq, score));
        } else {
            active.push((seq, score));
            saved += 1;
            if saved == beam_size {
                break;
            }
        }
    }
    (active, finished)
}

/// `BeamSearchDecoder.finalize` + `MaximumLikelihoodRanker.rank`: pad the
/// finished set with the best unfinished hypotheses (+ `<|endoftext|>`), then
/// pick the sequence with the highest `sum_logprob / sampled_len`. Returns the
/// winning full sequence (always eot-terminated), its sum, and whether it
/// ended on a genuinely sampled eot.
type RankedSeq = (Vec<u32>, f32, bool);

fn finalize_and_rank(
    active: &[Scored],
    finished: &[Scored],
    start_len: usize,
    beam_size: usize,
    eot: u32,
) -> Option<RankedSeq> {
    let mut candidates: Vec<RankedSeq> = finished
        .iter()
        .map(|(s, c)| (s.clone(), *c, true))
        .collect();
    if candidates.len() < beam_size {
        let mut ordered: Vec<_> = active.iter().collect();
        ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        for (seq, score) in ordered {
            if candidates.len() >= beam_size {
                break;
            }
            let mut padded = seq.clone();
            padded.push(eot);
            if !candidates.iter().any(|(s, _, _)| *s == padded) {
                candidates.push((padded, *score, false));
            }
        }
    }

    let mut best: Option<RankedSeq> = None;
    for (seq, score, term) in &candidates {
        if seq.len() <= start_len {
            continue;
        }
        let gen_len = seq.len().saturating_sub(start_len) - usize::from(seq.last() == Some(&eot));
        let rank = score / gen_len.max(1) as f32;
        if best.as_ref().is_none_or(|(_, bs, _)| rank > *bs) {
            best = Some((seq.clone(), *score, *term));
        }
    }
    best
}

/// The reference deterministic beam loop: expand each beam by its top
/// candidates, apply the same logit filters as greedy, stop once
/// `round(beam_size * patience)` sequences finish or the cap is hit, then
/// finalize and rank. The winning sequence (caller prefix + generated tokens +
/// a trailing `<|endoftext|>`) is written back into `tokens`.
pub fn beam_search<B: Backend>(
    whisper: &Whisper<B>,
    xa: &Tensor<B, 3>,
    tokens: &mut Vec<u32>,
    options: &BeamOptions,
) -> Result<BeamResult> {
    let n_vocab = whisper.dims.n_vocab;
    let sid = special_ids(n_vocab);
    let suppress = resolve_suppression(&options.suppress_tokens, n_vocab);
    let sot_index = tokens
        .iter()
        .position(|&t| t == sid.sot)
        .ok_or_else(|| Error::Tokenizer("beam search needs <|startoftranscript|>".into()))?;
    let dev = &xa.device();
    let beam_size = options.beam_size.max(1);
    let patience = if options.patience > 0.0 {
        options.patience
    } else {
        1.0
    };
    let max_candidates = ((beam_size as f32) * patience).round().max(1.0) as usize;
    let n_ctx = whisper.dims.n_text_ctx;
    let budget = options.max_tokens.max(1);

    let start_len = tokens.len();
    let prefix = tokens.clone();
    let mut beams: Vec<Hypothesis> = (0..beam_size)
        .map(|_| Hypothesis {
            tokens: prefix.clone(),
            sum_logprob: 0.0,
        })
        .collect();

    let mut finished: Vec<Scored> = Vec::new();
    let mut no_speech_prob = f32::NAN;

    for step in 0..budget {
        if beams.is_empty() {
            break;
        }
        let over_budget = beams[0].tokens.len() > n_ctx;
        if over_budget {
            break;
        }

        let mut step_logprobs: Vec<Vec<f32>> = Vec::with_capacity(beams.len());
        for beam in &beams {
            let ids: Vec<i64> = beam.tokens.iter().map(|&t| i64::from(t)).collect();
            let decoded = whisper.forward_decoder(
                Tensor::<B, 1, burn::tensor::Int>::from_data(
                    TensorData::new(ids, [beam.tokens.len()]),
                    dev,
                ),
                xa,
            ); // [seq, n_vocab]

            if step == 0 {
                // no-speech probability at the sot position, before filtering;
                // all beams share the initial tokens, so beam 0 is enough
                let sot_row = decoded
                    .clone()
                    .slice([sot_index..sot_index + 1, 0..n_vocab])
                    .reshape([n_vocab])
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| Error::Decoder(format!("reading sot logits: {e}")))?;
                no_speech_prob = log_softmax_row(&sot_row)[sid.nospeech as usize].exp();
            }

            let mut row = decoded
                .slice([beam.tokens.len() - 1..beam.tokens.len(), 0..n_vocab])
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
            step_logprobs.push(log_softmax_row(&row));
        }

        let (active, newly_finished) = select_candidates(
            expand_candidates(&beams, &step_logprobs, beam_size),
            beam_size,
            sid.eot,
        );

        // merge, capped at max_candidates, best scores first (the reference
        // `finished_sequences` dict)
        for (seq, score) in newly_finished {
            if finished.len() >= max_candidates {
                break;
            }
            if let Some(entry) = finished.iter_mut().find(|(s, _)| *s == seq) {
                entry.1 = score;
            } else {
                finished.push((seq, score));
            }
        }

        beams = active
            .into_iter()
            .map(|(tokens, sum_logprob)| Hypothesis {
                tokens,
                sum_logprob,
            })
            .collect();

        if finished.len() >= max_candidates {
            break;
        }
    }

    let active_debug: Vec<Scored> = beams
        .iter()
        .map(|h| (h.tokens.clone(), h.sum_logprob))
        .collect();
    let (winning, sum_logprob, terminated_eot) =
        finalize_and_rank(&active_debug, &finished, start_len, beam_size, sid.eot)
            .ok_or_else(|| Error::Decoder("beam search produced no candidate sequences".into()))?;

    let gen_len =
        winning.len().saturating_sub(start_len) - usize::from(winning.last() == Some(&sid.eot));
    let mean_logprob = sum_logprob / (gen_len + 1) as f32;

    tokens.clear();
    tokens.extend_from_slice(&winning);

    Ok(BeamResult {
        sum_logprob,
        mean_logprob,
        no_speech_prob,
        terminated_eot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::whisper::{EOT, SOT, TRANSCRIBE};

    fn hyp(tokens: &[u32], sum: f32) -> Hypothesis {
        Hypothesis {
            tokens: tokens.to_vec(),
            sum_logprob: sum,
        }
    }

    #[test]
    fn expand_keeps_beam_plus_one_candidates() {
        let beams = [hyp(&[SOT, TRANSCRIBE], 0.0)];
        // distinct logprobs so top-k is unambiguous; token 7 wins
        let row = vec![0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0];
        let expanded = expand_candidates(&beams, &[row], 1);
        assert_eq!(expanded.len(), 2, "beam_size+1 candidates");
        // top two tokens are 7 and 6, scored by upstream cumulative sum
        assert_eq!(expanded[0].0, vec![SOT, TRANSCRIBE, 7]);
        assert_eq!(expanded[1].0, vec![SOT, TRANSCRIBE, 6]);
        assert!(expanded[0].1 > expanded[1].1);
    }

    #[test]
    fn select_collects_eot_and_keeps_active() {
        let candidates = vec![
            (vec![1, 2, EOT], -0.5),
            (vec![1, 3], -0.8),
            (vec![1, 2], -1.0),
            (vec![1, 4, EOT], -1.2),
        ];
        let (active, finished) = select_candidates(candidates, 1, EOT);
        // the lone non-eot slot goes to the best non-eot; high-eot ranks as finished
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, vec![1, 3]);
        assert_eq!(finished.len(), 1, "iteration stops once the slot is filled");
        assert_eq!(finished[0].0, vec![1, 2, EOT]);
    }

    #[test]
    fn finalize_pads_short_finished_set() {
        let finished = vec![(vec![1, 9, EOT], -3.0)];
        let active = vec![(vec![1, 4, 4], -1.0), (vec![1, 5, 5, 5], -1.2)];
        let (best, sum, term) = finalize_and_rank(&active, &finished, 1, 2, EOT).unwrap();
        // padded (unterminated) candidate ranked in and wins on average score
        assert_eq!(best, vec![1, 4, 4, EOT]);
        assert_eq!(sum, -1.0);
        assert!(!term);
    }

    #[test]
    fn ranker_prefers_higher_average_logprob() {
        // both end on eot naturally; longer one has larger sum but worse average
        let finished = vec![(vec![1, 2, 2, 2, 2, 2, EOT], -4.0), (vec![1, 7, EOT], -1.1)];
        let (best, _, term) = finalize_and_rank(&[], &finished, 1, 2, EOT).unwrap();
        assert_eq!(best, vec![1, 7, EOT]);
        assert!(term);
    }

    #[test]
    fn beam1_selection_equals_argmax() {
        // a lone beam with top-2 expansion is just argmax of the row
        let beams = [hyp(&[1], 0.0)];
        let row = vec![0.0f32, -1.0, 5.0, 0.5, -2.0];
        let expanded = expand_candidates(&beams, &[log_softmax_row(&row)], 1);
        let (active, finished) = select_candidates(expanded, 1, EOT);
        assert!(finished.is_empty(), "eot not in row");
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, vec![1, 2]);
        // score = 0 (beam sum) + log_softmax of token 2 = -ln(sum(exp(row-5)))
        let expected =
            -(1.0f32 + (-5.0f32).exp() + (-6.0f32).exp() + (-4.5f32).exp() + (-7.0f32).exp()).ln();
        assert!(
            (active[0].1 - expected).abs() < 1e-4,
            "{} vs {expected}",
            active[0].1
        );
    }

    #[test]
    fn topk_returns_descending() {
        let row = vec![0.0f32, 3.0, 1.0, 2.0];
        let top = topk_logprobs(&row, 2);
        assert_eq!(top, vec![(3.0, 1), (2.0, 3)]);
    }
}
