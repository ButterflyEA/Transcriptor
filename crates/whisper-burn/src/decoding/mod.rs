//! Text-decoding helpers on top of the assembled model: language detection
//! first, greedy/beam token loops in later tasks.

pub mod beam;
pub mod greedy;
pub mod lang;
pub mod segments;

pub use beam::{BeamOptions, BeamResult, beam_search};
pub use greedy::{GreedyOptions, GreedyResult, greedy_search};
pub use lang::{
    LANGUAGE_BASE, LanguageInfo, SOT, detect_language, is_multilingual, language_window,
    num_languages,
};
pub use segments::{Segment, assemble_segments, timestamp_token_to_ms};
