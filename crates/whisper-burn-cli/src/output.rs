//! Transcript rendering for the CLI, shared with the desktop app via
//! `whisper_burn::format`.

pub use whisper_burn::format::{
    bracket_line, first_strong_is_rtl, format_json, format_srt, format_txt, format_vtt, srt_time,
    visual_order, vtt_time,
};