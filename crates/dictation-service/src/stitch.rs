//! Merges the text of consecutive overlapping windows.
use crate::transcript::normalize_word;

/// Appends `next` to `prev`, dropping the longest suffix of `prev` (at most
/// `max_overlap_words`) that equals a prefix of `next`.
pub fn stitch(prev: &[String], next: &[String], max_overlap_words: usize) -> Vec<String> {
    let max = max_overlap_words.min(prev.len()).min(next.len());
    let overlap = (1..=max)
        .rev()
        .find(|&k| {
            prev[prev.len() - k..]
                .iter()
                .zip(&next[..k])
                .all(|(a, b)| normalize_word(a) == normalize_word(b))
        })
        .unwrap_or(0);
    let mut out = prev[..prev.len() - overlap].to_vec();
    out.extend(next.iter().cloned());
    out
}