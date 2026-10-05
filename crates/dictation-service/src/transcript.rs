//! Pure state machine for streaming dictation: locked text, plus a tentative
//! tail that later passes may correct with backspaces.

/// Removes punctuation, Hebrew niqqud and cantillation, and lowercases.
pub fn normalize_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric() && !('\u{0591}'..='\u{05C7}').contains(c))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(String::from).collect()
}

/// Number of leading words that are equal after normalisation.
pub fn agreed_prefix_len(a: &[String], b: &[String]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(x, y)| normalize_word(x) == normalize_word(y))
        .count()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    Append(String),
    Replace { backspaces: usize, text: String },
}

pub struct Transcript {
    locked: Vec<String>,
    typed: Vec<String>,
    last: Vec<String>,
    lock_words: usize,
    append_only: bool,
}

impl Transcript {
    pub fn new(lock_words: usize) -> Self {
        Self {
            locked: Vec::new(),
            typed: Vec::new(),
            last: Vec::new(),
            lock_words,
            append_only: false,
        }
    }

    pub fn locked_words(&self) -> usize {
        self.locked.len()
    }

    pub fn locked_text(&self) -> String {
        self.locked.join(" ")
    }

    /// Stop correcting from now on; everything typed stays as it is.
    pub fn set_append_only(&mut self) {
        self.append_only = true;
    }

    /// Feeds a pass over the audio that is not locked yet. Returns at most one edit.
    pub fn on_pass(&mut self, pass_text: &str) -> Option<Edit> {
        let w = words(pass_text);
        let edit = self.edit_to(&w);
        let agreed = agreed_prefix_len(&self.last, &w);
        let n = agreed
            .min(w.len().saturating_sub(self.lock_words))
            .min(self.typed.len());
        self.locked.extend(self.typed.drain(..n));
        self.last = w[n.min(w.len())..].to_vec();
        edit
    }

    /// Final pass: returns the closing edit and the full final text.
    pub fn finish(&mut self, final_text: &str) -> (Option<Edit>, String) {
        let w = words(final_text);
        let edit = self.edit_to(&w);
        let mut all = self.locked.clone();
        all.extend(self.typed.iter().cloned());
        (edit, all.join(" "))
    }

    fn render(&self, words: &[String]) -> String {
        let lead = if self.locked.is_empty() { "" } else { " " };
        if words.is_empty() {
            String::new()
        } else {
            format!("{lead}{}", words.join(" "))
        }
    }

    fn edit_to(&mut self, w: &[String]) -> Option<Edit> {
        if self.append_only {
            if w.len() <= self.typed.len() {
                return None;
            }
            let extra = &w[self.typed.len()..];
            let lead = if self.locked.is_empty() && self.typed.is_empty() {
                ""
            } else {
                " "
            };
            let text = format!("{lead}{}", extra.join(" "));
            self.typed.extend(extra.iter().cloned());
            return Some(Edit::Append(text));
        }
        let old = self.render(&self.typed);
        let new = self.render(w);
        self.typed = w.to_vec();
        let common = old
            .chars()
            .zip(new.chars())
            .take_while(|(a, b)| a == b)
            .count();
        let backspaces = old.chars().count() - common;
        let text: String = new.chars().skip(common).collect();
        match (backspaces, text.is_empty()) {
            (0, true) => None,
            (0, false) => Some(Edit::Append(text)),
            _ => Some(Edit::Replace { backspaces, text }),
        }
    }
}
