//! Reference `ApplyTimestampRules` logit filter, ported from
//! `whisper/decoding.py`.
//!
//! Applied to a logits row *before* argmax (greedy) or `log_softmax` (beam),
//! once per hypothesis, exactly as the reference `LogitFilter` does.
//! `sampled` is the generated-token suffix only: the caller-supplied prefix
//! (sot, language, task, `<|startofprev|>`) must not count toward the rules.

use crate::tokenizer::whisper::TextTokenizer;

/// Reference `max_initial_timestamp` default (1.0 s) expressed in timestamp
/// steps: `round(1.0 / 0.02) == 50`.
pub const DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX: u32 = 50;

/// log_softmax with torch semantics (subtract the running max). A row that is
/// entirely `-inf` maps to `-inf` rather than NaN.
fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; row.len()];
    }
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    row.iter().map(|v| v - max - sum.ln()).collect()
}

/// Mask `logits` in place with the reference timestamp rules.
///
/// Mirrors `ApplyTimestampRules.apply`:
/// 1. suppress `<|notimestamps|>`;
/// 2. timestamps arrive in adjacent pairs, so after a completed pair text is
///    required and after an opened pair a closing timestamp is required;
/// 3. timestamps must not decrease and every segment must be non-empty;
/// 4. the first sampled token must be a timestamp, capped by
///    `max_initial_timestamp_index`;
/// 5. if the summed log-probability over all timestamps beats the best text
///    token, suppress text.
pub fn apply_timestamp_rules(
    logits: &mut [f32],
    sampled: &[u32],
    tokenizer: &TextTokenizer,
    max_initial_timestamp_index: Option<u32>,
) {
    let ts = tokenizer.timestamp_begin as usize;
    let eot = tokenizer.special.eot as usize;
    let neg = f32::NEG_INFINITY;
    let n = logits.len();

    // 1. suppress <|notimestamps|>
    let notimestamps = tokenizer.special.notimestamps as usize;
    if notimestamps < n {
        logits[notimestamps] = neg;
    }

    let last_was_timestamp = sampled.last().is_some_and(|&t| t >= tokenizer.timestamp_begin);
    let penultimate_was_timestamp = match sampled.len() {
        0 | 1 => true,
        n => sampled[n - 2] >= tokenizer.timestamp_begin,
    };

    // 2. pair discipline
    if last_was_timestamp {
        if penultimate_was_timestamp {
            // a pair just closed: text must follow, so no new timestamp yet
            if ts < n {
                logits[ts..].fill(neg);
            }
        } else {
            // a pair just opened: a closing timestamp is owed, so no text
            logits[..eot.min(n)].fill(neg);
        }
    }

    // 3. monotonic, non-empty segments
    if let Some(&last_ts) = sampled
        .iter()
        .rev()
        .find(|&&t| t >= tokenizer.timestamp_begin)
    {
        let limit = if last_was_timestamp && !penultimate_was_timestamp {
            last_ts as usize
        } else {
            last_ts as usize + 1
        };
        let limit = limit.min(n);
        if limit > ts {
            logits[ts..limit].fill(neg);
        }
    }

    // 4. the first sampled token must be a bounded timestamp
    if sampled.is_empty() {
        logits[..ts.min(n)].fill(neg);
        if let Some(index) = max_initial_timestamp_index {
            let last_allowed = (ts + index as usize).min(n.saturating_sub(1));
            logits[last_allowed + 1..].fill(neg);
        }
    }

    // 5. summed timestamp log-probability versus best text log-probability
    if ts < n {
        let logprobs = log_softmax(logits);
        let timestamp_logprob = logprobs[ts..]
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.exp())
            .sum::<f32>()
            .ln();
        let max_text_logprob = logprobs[..ts]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if timestamp_logprob > max_text_logprob {
            logits[..ts].fill(neg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::whisper::TextTokenizer;

    const N_VOCAB: u32 = 51865;

    /// The v2 multilingual layout: `n_vocab = 51865`, timestamps from 50364.
    /// `standard` is fallible, hence the `unwrap`.
    fn tk() -> TextTokenizer {
        TextTokenizer::standard(N_VOCAB, 1500).unwrap()
    }

    /// A row where one text token dominates. This matters: rule 5 compares the
    /// *summed* log-probability of all ~1500 timestamp tokens against the
    /// *best* single text token, so on a flat row timestamps always win and
    /// text is always suppressed - which would mask the very thing each test
    /// is trying to isolate.
    fn text_favored(text_token: u32) -> Vec<f32> {
        let mut row = vec![0.0f32; N_VOCAB as usize];
        row[text_token as usize] = 20.0;
        row
    }

    fn is_masked(row: &[f32], token: u32) -> bool {
        row[token as usize] == f32::NEG_INFINITY
    }

    #[test]
    fn first_token_must_be_a_timestamp() {
        let t = tk();
        let mut row = text_favored(1000);
        apply_timestamp_rules(
            &mut row,
            &[],
            &t,
            Some(DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX),
        );
        assert!(
            is_masked(&row, 1000),
            "text masked before the first timestamp"
        );
        assert!(is_masked(&row, t.special.notimestamps));
    }

    #[test]
    fn first_timestamp_is_capped_by_max_initial() {
        let t = tk();
        let mut row = text_favored(1000);
        apply_timestamp_rules(
            &mut row,
            &[],
            &t,
            Some(DEFAULT_MAX_INITIAL_TIMESTAMP_INDEX),
        );
        // default cap is 1.0 s == 50 steps; 51 is one step too late
        assert!(!is_masked(&row, t.timestamp_begin + 50), "cap allows 1.00s");
        assert!(is_masked(&row, t.timestamp_begin + 51), "beyond 1.00s");
    }

    #[test]
    fn a_lone_leading_timestamp_counts_as_a_closed_pair() {
        let t = tk();
        // Reference quirk, preserved deliberately: `penultimate_was_timestamp`
        // is `len(tokens) < 2 or ...`, so with only one sampled token the pair
        // reads as CLOSED and text is allowed. Only once some text precedes a
        // timestamp does the pair read as open. This is what lets whisper
        // emit `ts text ts text ...` rather than `ts ts text ts ts text`.
        let mut row = text_favored(1000);
        apply_timestamp_rules(&mut row, &[t.timestamp_begin + 5], &t, None);
        assert!(is_masked(&row, t.timestamp_begin + 6), "new ts blocked");
        assert!(!is_masked(&row, 1000), "text allowed after one timestamp");
    }

    #[test]
    fn opened_pair_forces_a_closing_timestamp() {
        let t = tk();
        // text preceded this timestamp, so the pair is genuinely open and a
        // closing timestamp is owed
        let mut row = text_favored(1000);
        apply_timestamp_rules(&mut row, &[1000, t.timestamp_begin + 5], &t, None);
        assert!(
            is_masked(&row, 1000),
            "text masked while the pair is open"
        );
        assert!(!is_masked(&row, t.timestamp_begin + 6), "a later ts closes it");
    }

    #[test]
    fn closed_pair_forbids_immediate_new_timestamp() {
        let t = tk();
        // two adjacent timestamps closed the pair, so text must follow
        let sampled = [t.timestamp_begin + 5, t.timestamp_begin + 20];
        let mut row = text_favored(1000);
        apply_timestamp_rules(&mut row, &sampled, &t, None);
        assert!(is_masked(&row, t.timestamp_begin + 21), "new ts blocked");
        assert!(!is_masked(&row, 1000), "text allowed after a closed pair");
    }

    #[test]
    fn timestamps_never_decrease() {
        let t = tk();
        // The pair opened at +20.00s, so the closing timestamp may not sit
        // before it. Text precedes the open, so rule 2 masks text rather than
        // timestamps, leaving rule 3's effect observable.
        let sampled = [t.timestamp_begin + 5, 1000, t.timestamp_begin + 400];
        let mut row = text_favored(1000);
        apply_timestamp_rules(&mut row, &sampled, &t, None);
        assert!(is_masked(&row, t.timestamp_begin + 399), "decrease blocked");
        assert!(!is_masked(&row, t.timestamp_begin + 400), "equal allowed");
    }

    #[test]
    fn notimestamps_is_always_suppressed() {
        let t = tk();
        for sampled in [vec![], vec![t.timestamp_begin], vec![1000]] {
            let mut row = text_favored(1000);
            apply_timestamp_rules(&mut row, &sampled, &t, None);
            assert!(
                is_masked(&row, t.special.notimestamps),
                "notimestamps must always be masked"
            );
        }
    }

    #[test]
    fn summed_timestamp_prob_forces_a_timestamp_mid_text() {
        let t = tk();
        // mid-text, a text token dominates, so the rules leave text alone
        let mut favored = text_favored(1000);
        apply_timestamp_rules(&mut favored, &[1000], &t, None);
        assert!(!is_masked(&favored, 1000), "dominant text survives");

        // on a flat row the ~1500 timestamps outweigh any single text token,
        // so the reference forces a timestamp here
        let mut flat = vec![0.0f32; N_VOCAB as usize];
        apply_timestamp_rules(&mut flat, &[1000], &t, None);
        assert!(
            is_masked(&flat, 1000),
            "summed timestamp prob must suppress text"
        );
    }
}
