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

#[test]
fn json_serializes_segments() {
    let out = format_json(&[seg(0, 45000, "Hello")]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v[0]["start"], serde_json::json!(0.0));
    assert_eq!(v[0]["end"], serde_json::json!(45.0));
    assert_eq!(v[0]["text"], serde_json::json!("Hello"));
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