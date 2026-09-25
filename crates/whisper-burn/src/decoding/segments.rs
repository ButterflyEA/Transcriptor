//! Reference-format timestamped segment assembly.
//!
//! A decoded window is a stream of `<|startoftranscript|>`, task/language
//! specials, `<|timestamp|>` markers and text tokens. Text runs between two
//! timestamp tokens become `Segment`s: `start = seek + first_timestamp_ms`,
//! `end = seek + next_timestamp_ms`; timer values derive purely from the token
//! ids (`(token - timestamp_begin) * 10 ms`, whisper's `time_precision`).
//!
//! Reference whisper emits timestamps as adjacent pairs surrounding each text
//! run (`<|0.00|> <|4.00|> text <|4.00|> <|8.00|> text...`), so splitting at
//! every timestamp boundary coincides with the reference segmentation. Runs
//! with no closing timestamp (leading text before the first timestamp, or a
//! tail past the last one) are dropped, mirroring the reference, which only
//! emits timestamp-bounded slices. Empty runs between consecutive timestamps
//! emit nothing.
//!
//! Deviation from the plan sketch: `assemble_segments` returns `Vec<Segment>`
//! (no `avg_logprob`) — that statistic comes from the decoding result, not
//! from the token stream, and `sample_rate` is not needed because times derive
//! purely from tokens.

use crate::Result;
use crate::tokenizer::whisper::TextTokenizer;

/// Milliseconds per timestamp step (whisper `time_precision =
/// hop_length / sample_rate * 1000` = 10 ms).
pub const TIME_PRECISION_MS: u32 = 10;

/// A transcribed text segment with millisecond start/end times (window
/// offsets already applied, i.e. relative to the original audio).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start: u32,
    pub end: u32,
    pub text: String,
}

/// Convert a `<|X.XX|>` token id to milliseconds: `(token - timestamp_begin)
/// * TIME_PRECISION_MS`. Non-timestamp tokens saturate to 0.
pub fn timestamp_token_to_ms(token: u32, tokenizer: &TextTokenizer) -> u32 {
    token.saturating_sub(tokenizer.timestamp_begin) * TIME_PRECISION_MS
}

/// Split a decoded window token stream into timestamp-bounded segments.
/// `seek` is the window's start time (in ms) in the original audio and is
/// added to every segment time. Text tokens (`< eot`) accumulate between
/// timestamps; specials (`eot..timestamp_begin`) are ignored (sot sequence,
/// language/task markers, eot).
pub fn assemble_segments(
    tokens: &[u32],
    tokenizer: &TextTokenizer,
    seek: u32,
) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut open_ms: Option<u32> = None;
    for &t in tokens {
        if tokenizer.is_timestamp(t) {
            let ms = seek + timestamp_token_to_ms(t, tokenizer);
            if let Some(start) = open_ms
                && !text_tokens.is_empty()
            {
                let text = tokenizer.decode(&text_tokens)?;
                segments.push(Segment {
                    start,
                    end: ms,
                    text,
                });
            }
            open_ms = Some(ms);
            text_tokens.clear();
        } else if t < tokenizer.eot() {
            text_tokens.push(t);
        }
    }
    Ok(segments)
}
