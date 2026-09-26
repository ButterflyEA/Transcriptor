use std::io::Read;
use whisper_burn::format::{format_docx, format_json, format_pdf, format_srt, format_txt, format_vtt};
use whisper_burn::TranscriptionSegment;

fn seg(start: u32, end: u32, text: &str) -> TranscriptionSegment {
    TranscriptionSegment { start, end, text: text.to_string() }
}

#[test]
fn txt_is_plain_logical_lines() {
    let out = format_txt(&[seg(0, 2000, "שלום עולם"), seg(2000, 5000, "hello")]);
    assert_eq!(out, "שלום עולם\nhello\n");
}

/// The reference JSON is an object, not a bare array: a `language` header plus
/// `segments`. See `docs/golden/reference_large-v3_auto_golden.json`, whose
/// only top-level keys are `language` and `segments` and whose only segment
/// keys are `start`, `end` and `text`.
#[test]
fn json_matches_the_reference_schema() {
    let out = format_json(
        "he",
        &[seg(0, 45_000, "שלום"), seg(45_000, 48_640, "עולם")],
    );
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();

    assert_eq!(v["language"], serde_json::json!("he"));
    let segments = v["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0]["start"], serde_json::json!("00:00.000"));
    assert_eq!(segments[0]["end"], serde_json::json!("00:45.000"));
    assert_eq!(segments[0]["text"], serde_json::json!("שלום"));
    assert_eq!(segments[1]["start"], serde_json::json!("00:45.000"));

    // the segment objects carry nothing but start/end/text. Compared as a set:
    // serde_json orders object keys alphabetically, and key order is not part
    // of the contract - golden comparison is structural.
    for segment in segments {
        let mut keys: Vec<&str> = segment
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["end", "start", "text"], "unexpected keys: {keys:?}");
    }
}

/// Timestamps are `MM:SS.mmm` with *unbounded* minutes - the golden writes
/// `00:07.800`, not `00:00:07.800`, so there is no hours field. 75 minutes is
/// therefore `75:00.000`.
///
/// This contradicts the "HH:MM:SS.mmm" shape originally written into the
/// design doc; the committed golden is the authority, and this test exists to
/// keep the two from drifting apart again.
#[test]
fn json_timestamps_are_minutes_seconds_milliseconds() {
    let out = format_json("en", &[seg(0, 4_500_000, "long")]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["segments"][0]["start"], serde_json::json!("00:00.000"));
    assert_eq!(v["segments"][0]["end"], serde_json::json!("75:00.000"));
}

#[test]
fn srt_has_numbered_cues() {
    let out = format_srt(&[seg(1000, 2500, "Hi")]);
    assert!(out.starts_with("1\n00:00:01,000 --> 00:00:02,500\nHi\n"), "{out}");
}

#[test]
fn vtt_has_webvtt_header() {
    let out = format_vtt(&[seg(1000, 2500, "Hi")]);
    assert!(out.starts_with("WEBVTT\n"), "{out}");
    assert!(out.contains("00:01.000 --> 00:02.500"), "{out}");
}

#[test]
fn docx_embeds_segment_text_and_timestamps() {
    let segs = [seg(0, 2500, "שלום עולם")];
    let bytes = format_docx(&segs, true).unwrap();
    assert!(bytes.len() > 100, "unreasonably small docx");
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let mut xml = String::new();
    zip.by_name("word/document.xml").unwrap().read_to_string(&mut xml).unwrap();
    assert!(xml.contains("[00:00.000 --&gt; 00:02.500]"), "timestamps missing:\n{xml}");
    assert!(xml.contains("שלום עולם"), "segment text missing:\n{xml}");
}

#[test]
fn pdf_produces_a_structural_pdf() {
    let segs = [seg(0, 2500, "שלום עולם")];
    let bytes = format_pdf(&segs, true).unwrap();
    assert!(bytes.starts_with(b"%PDF"), "not a PDF header");
    assert!(bytes.len() > 500, "unreasonably small pdf");
}