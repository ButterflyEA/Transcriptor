//! Format-output rendering tests. These exercise the `output` module of the
//! CLI library (binaries alone can't be imported from integration tests).

use whisper_burn::TranscriptionSegment;
use whisper_burn_cli::output;

fn seg(start: u32, end: u32, text: &str) -> TranscriptionSegment {
    TranscriptionSegment {
        start,
        end,
        text: text.to_string(),
    }
}

#[test]
fn srt_time_is_clock_with_comma_millis() {
    assert_eq!(output::srt_time(500), "00:00:00,500");
    assert_eq!(output::srt_time(65_000), "00:01:05,000");
    assert_eq!(output::srt_time(3_600_000), "01:00:00,000");
}

#[test]
fn vtt_time_is_minutes_clock_with_dot_millis() {
    assert_eq!(output::vtt_time(500), "00:00.500");
    assert_eq!(output::vtt_time(65_000), "01:05.000");
}

#[test]
fn bracket_line_matches_whisper_console_style() {
    assert_eq!(
        output::bracket_line(&seg(0, 500, "hello")),
        "[00:00.000 --> 00:00.500]  hello"
    );
}

#[test]
fn txt_renders_plain_text_lines() {
    let segs = [seg(0, 400, "And so"), seg(500, 900, "my fellow Americans")];
    assert_eq!(output::format_txt(&segs), "And so\nmy fellow Americans\n");
}

#[test]
fn srt_renders_cue_blocks() {
    let segs = [seg(0, 400, "And so"), seg(500, 900, "my fellow Americans")];
    let expected = "\
1
00:00:00,000 --> 00:00:00,400
And so

2
00:00:00,500 --> 00:00:00,900
my fellow Americans
";
    assert_eq!(output::format_srt(&segs), expected);
}

#[test]
fn vtt_renders_webvtt_blocks() {
    let segs = [seg(0, 400, "And so")];
    let expected = "\
WEBVTT

00:00.000 --> 00:00.400
And so";
    assert_eq!(output::format_vtt(&segs), expected);
}

#[test]
fn json_renders_seconds_and_text() {
    let segs = [seg(0, 500, "hello")];
    let s = output::format_json(&segs);
    let v: serde_json::Value = serde_json::from_str(&s).expect("json parses");
    let arr = v.as_array().expect("json is an array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["start"], serde_json::json!(0.0));
    assert_eq!(arr[0]["end"], serde_json::json!(0.5));
    assert_eq!(arr[0]["text"], serde_json::json!("hello"));
}

#[test]
fn visual_order_mirrors_hebrew_and_keeps_ltr() {
    // a bidi-less terminal draws glyphs left-to-right, so RTL text must be
    // emitted mirrored (visual order) to read correctly on screen
    assert_eq!(output::visual_order("שלום"), "םולש");
    assert_eq!(
        output::visual_order("שלום, זהו ניסוי"),
        "יוסינ והז ,םולש"
    );
    assert_eq!(output::visual_order("2 עגבניות ושום"), "םושו תוינבגע 2");
    // LTR text is never mirrored
    assert_eq!(output::visual_order("hello"), "hello");
    assert_eq!(output::visual_order("(2:30) a.m."), "(2:30) a.m.");
}

#[test]
fn bracket_line_renders_hebrew_in_visual_order_for_bidi_less_terminals() {
    // VS Code's terminal (like classic conhost) runs no bidirectional
    // algorithm, so RTL segment text is emitted mirrored for the console
    let line = output::bracket_line(&seg(0, 500, "שלום, זהו ניסוי"));
    assert_eq!(line, "[00:00.000 --> 00:00.500]  יוסינ והז ,םולש");
    // LTR text stays untouched
    assert_eq!(
        output::bracket_line(&seg(0, 500, "(2:30) a.m.")),
        "[00:00.000 --> 00:00.500]  (2:30) a.m."
    );
}

#[test]
fn file_outputs_keep_hebrew_in_logical_order_without_marks() {
    // files are consumed by bidi-aware renderers, so they stay in logical
    // order and carry no invisible direction marks
    let segs = [seg(0, 400, "שלום, זהו ניסוי")];
    assert_eq!(output::format_txt(&segs), "שלום, זהו ניסוי\n");
    assert!(!output::format_txt(&segs).contains('\u{200f}'));

    let srt = output::format_srt(&segs);
    assert!(!srt.contains('\u{200f}'), "srt must be mark-free: {srt:?}");
    assert!(
        srt.contains("\nשלום, זהו ניסוי"),
        "srt must keep logical cue text: {srt:?}"
    );

    let vtt = output::format_vtt(&segs);
    assert!(!vtt.contains('\u{200f}'), "vtt must be mark-free: {vtt:?}");
    assert!(
        vtt.contains("\nשלום, זהו ניסוי"),
        "vtt must keep logical cue text: {vtt:?}"
    );
}

#[test]
fn json_keeps_raw_text_without_direction_marks() {
    let segs = [seg(0, 500, "שלום")];
    let s = output::format_json(&segs);
    assert!(
        !s.contains('\u{200f}'),
        "json must be pure data (no direction marks): {s:?}"
    );
    assert!(s.contains("שלום"), "json lost the text: {s:?}");
    assert!(
        !s.contains("םולש"),
        "json must stay logical (no visual order): {s:?}"
    );
}
