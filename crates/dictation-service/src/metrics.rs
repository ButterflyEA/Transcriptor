//! Text comparison helpers for the streaming measurements.

/// Lowercases, drops punctuation and Hebrew niqqud/cantillation, and collapses
/// whitespace, so only the words themselves are compared.
pub fn normalize_for_compare(s: &str) -> String {
    let kept: String = s
        .chars()
        .filter(|c| !('\u{0591}'..='\u{05C7}').contains(c))
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .flat_map(char::to_lowercase)
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Levenshtein distance over chars.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Character error rate of `hyp` against `reference` after normalisation.
pub fn char_error_rate(reference: &str, hyp: &str) -> f64 {
    let r = normalize_for_compare(reference);
    let h = normalize_for_compare(hyp);
    let n = r.chars().count();
    if n == 0 {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(&r, &h) as f64 / n as f64
}