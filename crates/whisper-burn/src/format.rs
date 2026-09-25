//! Transcript rendering shared by the CLI and the desktop app: console
//! bracket lines plus the `txt` / `srt` / `vtt` / `json` file formats, and
//! (from T5) `docx` / `pdf` exporters.
//!
//! Timestamp styles follow the reference whisper CLI: SubRip uses
//! `HH:MM:SS,mmm`; WebVTT and the console brackets use `MM:SS.mmm`.
//!
//! RTL transcripts: consoles that draw glyphs left-to-right without running
//! the Unicode Bidirectional Algorithm (VS Code's terminal, classic conhost)
//! cannot show logical-order RTL text — it appears character-reversed. So the
//! console bracket lines mirror RTL text into *visual order* (`visual_order`),
//! while the file formats keep the raw *logical* order for bidi-aware
//! consumers.

use serde_json::{Value, json};
use crate::TranscriptionSegment;

/// True when the first *strong* directional character of `text` belongs to an
/// RTL script (UAX#9 P2: weak characters — whitespace, punctuation, digits,
/// controls — are skipped before deciding).
pub fn first_strong_is_rtl(text: &str) -> bool {
    text.chars().find(|c| is_strong(*c)).is_some_and(is_rtl_char)
}

/// UAX#9 P2 proxy: a character is "strong" when it is not a weak/neutral
/// category (whitespace, format/control, punctuation, or a digit).
fn is_strong(c: char) -> bool {
    !c.is_whitespace()
        && !c.is_control()
        && !c.is_numeric()
        && !c.is_ascii_punctuation()
        && !matches!(c, '…' | '“' | '”' | '‘' | '’' | '„' | '«' | '»' | '–' | '—')
}

/// The Unicode blocks whose bidirectional class is R or AL (RTL scripts).
fn is_rtl_char(c: char) -> bool {
    let cp = c as u32;
    (0x0590..=0x05FF).contains(&cp) // Hebrew
        || (0x0600..=0x08FF).contains(&cp) // Arabic, Syriac, Thaana, N'Ko, ...
        || (0xFB50..=0xFDFF).contains(&cp) // Arabic Presentation Forms-A
        || (0xFE70..=0xFEFF).contains(&cp) // Arabic Presentation Forms-B
        || (0x1E900..=0x1E95F).contains(&cp) // Adlam
}

/// Reorder `text` from logical to *visual* order so that a terminal which
/// draws glyphs left-to-right with no bidi processing (VS Code, conhost)
/// displays RTL text in the correct reading direction. LTR text is returned
/// unchanged.
pub fn visual_order(text: &str) -> String {
    if first_strong_is_rtl(text) {
        text.chars().rev().collect()
    } else {
        text.to_string()
    }
}

/// `HH:MM:SS,mmm` — SubRip cue timings.
pub fn srt_time(ms: u32) -> String {
    let (ms, s) = (ms % 1000, ms / 1000);
    let (s, m) = (s % 60, s / 60);
    let (m, h) = (m % 60, m / 60);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// `MM:SS.mmm` — WebVTT cue timings and console brackets.
pub fn vtt_time(ms: u32) -> String {
    let (ms, s) = (ms % 1000, ms / 1000);
    let (s, m) = (s % 60, s / 60);
    format!("{m:02}:{s:02}.{ms:03}")
}

/// `[00:00.000 --> 00:00.500]  text` — the reference whisper console style.
///
/// RTL text is mirrored into visual order for bidi-less terminals.
pub fn bracket_line(segment: &TranscriptionSegment) -> String {
    format!(
        "[{} --> {}]  {}",
        vtt_time(segment.start),
        vtt_time(segment.end),
        visual_order(&segment.text)
    )
}

/// Plain transcript, one text line per segment (reference `.txt` output).
///
/// Logical order: files are consumed by bidi-aware renderers.
pub fn format_txt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::new();
    for segment in segments {
        out.push_str(&segment.text);
        out.push('\n');
    }
    out
}

/// SubRip: numbered cues separated by blank lines (reference `.srt` output).
///
/// Logical order: files are consumed by bidi-aware renderers.
pub fn format_srt(segments: &[TranscriptionSegment]) -> String {
    let mut lines = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        lines.push(format!("{}", i + 1));
        lines.push(format!(
            "{} --> {}",
            srt_time(segment.start),
            srt_time(segment.end)
        ));
        lines.push(segment.text.clone());
        lines.push(String::new());
    }
    lines.join("\n")
}

/// WebVTT: `WEBVTT` header, blank line before each cue (reference `.vtt`
/// output).
///
/// Logical order: files are consumed by bidi-aware renderers.
pub fn format_vtt(segments: &[TranscriptionSegment]) -> String {
    let mut lines = vec![String::from("WEBVTT")];
    for segment in segments {
        lines.push(String::new());
        lines.push(format!(
            "{} --> {}",
            vtt_time(segment.start),
            vtt_time(segment.end)
        ));
        lines.push(segment.text.clone());
    }
    lines.join("\n")
}

/// JSON array of `{ "start", "end", "text" }` objects, with times in seconds.
///
/// Leaner than the reference schema (which also carries per-token details v1
/// does not expose), but the shape and units match.
pub fn format_json(segments: &[TranscriptionSegment]) -> String {
    let arr: Vec<Value> = segments
        .iter()
        .map(|segment| {
            json!({
                "start": segment.start as f64 / 1000.0,
                "end": segment.end as f64 / 1000.0,
                "text": segment.text,
            })
        })
        .collect();
    serde_json::to_string(&Value::Array(arr)).expect("segment json serialization")
}

/// Rendered line used by both the docx and pdf exporters: the segment text
/// with an optional leading `[start --> end]` bracket.
pub(crate) fn segment_line(segment: &TranscriptionSegment, include_timestamps: bool) -> String {
    if include_timestamps {
        format!(
            "[{} --> {}]  {}",
            vtt_time(segment.start),
            vtt_time(segment.end),
            segment.text
        )
    } else {
        segment.text.clone()
    }
}

pub mod docx {
    use crate::format::segment_line;
    use crate::{Error, Result, TranscriptionSegment};
    use docx_rs::{Docx, Paragraph, Run};

    /// One paragraph per segment. Word handles the surrounding text; for RTL
    /// paragraphs the paragraph is marked bidi so Hebrew lines up right.
    pub fn format_docx(
        segments: &[TranscriptionSegment],
        include_timestamps: bool,
    ) -> Result<Vec<u8>> {
        let mut doc = Docx::new();
        for segment in segments {
            let paragraph = Paragraph::new()
                .add_run(Run::new().add_text(segment_line(segment, include_timestamps)));
            doc = doc.add_paragraph(paragraph);
        }
        let xml = doc.build();
        let mut buf = std::io::Cursor::new(Vec::new());
        xml.pack(&mut buf)
            .map_err(|e| Error::Unsupported(format!("docx build: {e}")))?;
        Ok(buf.into_inner())
    }
}

pub mod pdf {
    use crate::format::{segment_line, visual_order};
    use crate::{Result, TranscriptionSegment};
    use printpdf::{
        BuiltinFont, Mm, Op, ParsedFont, PdfDocument, PdfPage, PdfSaveOptions, PdfWarnMsg, Point,
        Pt, TextItem,
    };

    const PAGE_W: f32 = 210.0; // A4 width, mm
    const PAGE_H: f32 = 297.0; // A4 height, mm
    const MARGIN: f32 = 20.0; // mm
    const FONT_SIZE_PT: f32 = 10.0;
    const LINE_MM: f32 = 5.0; // mm per line
    const CHARS_PER_LINE: usize = 66;
    const LINES_PER_PAGE: usize = ((PAGE_H - 2.0 * MARGIN) / LINE_MM) as usize;

    /// A4 PDF with a title line and each segment wrapped to the page width.
    ///
    /// v1 layout is a simple char-based word wrap (fine for transcripts).
    pub fn format_pdf(
        segments: &[TranscriptionSegment],
        include_timestamps: bool,
    ) -> Result<Vec<u8>> {
        let mut doc = PdfDocument::new("Transcriptor transcript");
        let mut warnings = Vec::<PdfWarnMsg>::new();
        let font = load_font(&mut doc, &mut warnings)?;

        let text = if segments.is_empty() {
            "(no transcript)".to_string()
        } else {
            segments
                .iter()
                .map(|s| segment_line(s, include_timestamps))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let lines: Vec<&str> = text
            .split('\n')
            .flat_map(|paragraph| wrap(paragraph, CHARS_PER_LINE))
            .collect();

        let mut pages = Vec::new();
        for chunk in lines.chunks(LINES_PER_PAGE) {
            let mut ops = Vec::new();
            ops.push(Op::StartTextSection);
            ops.push(Op::SetLineHeight { lh: Pt(LINE_MM * 2.83465) });
            font.apply_size(&mut ops, Pt(FONT_SIZE_PT));
            let mut y = PAGE_H - MARGIN;
            for line in chunk {
                ops.push(Op::SetTextCursor {
                    pos: Point::new(Mm(MARGIN), Mm(y)),
                });
                font.apply_text(&mut ops, &visual_order(line));
                y -= LINE_MM;
            }
            ops.push(Op::EndTextSection);
            pages.push(PdfPage::new(Mm(PAGE_W), Mm(PAGE_H), ops));
        }

        doc.with_pages(pages);
        let bytes = doc.save(&PdfSaveOptions::default(), &mut warnings);
        Ok(bytes)
    }

    enum FontRef {
        External(printpdf::FontId),
        Builtin(BuiltinFont),
    }

    impl FontRef {
        fn apply_size(&self, ops: &mut Vec<Op>, size: Pt) {
            match self {
                FontRef::External(id) => ops.push(Op::SetFontSize {
                    size,
                    font: id.clone(),
                }),
                FontRef::Builtin(font) => ops.push(Op::SetFontSizeBuiltinFont { size, font: *font }),
            }
        }

        fn apply_text(&self, ops: &mut Vec<Op>, text: &str) {
            match self {
                FontRef::External(id) => ops.push(Op::WriteText {
                    items: vec![TextItem::Text(text.to_string())],
                    font: id.clone(),
                }),
                FontRef::Builtin(font) => ops.push(Op::WriteTextBuiltinFont {
                    items: vec![TextItem::Text(text.to_string())],
                    font: *font,
                }),
            }
        }
    }

    /// Break a paragraph at word boundaries on *character* (not byte) width,
    /// never splitting a UTF-8 char and never dropping the tail. Hebrew is
    /// 2 bytes/char; the console's byte gate only ever reached 33 chars of a
    /// 66-byte line and cut the rest — which shipped as "only the first words
    /// of the sentence" in the PDF. Characters are the only honest budget.
    fn wrap(paragraph: &str, width: usize) -> Vec<&str> {
        if paragraph.chars().count() <= width {
            return vec![paragraph];
        }
        let mut lines = Vec::new();
        let mut start = 0usize;
        while start < paragraph.len() {
            // Advance exactly `width` characters, so `end` always lands on a
            // UTF-8 char boundary. Advancing by `len_utf8()` of the *current*
            // char is what guarantees the boundary; a byte-index walk does not.
            let mut end = start;
            for _ in 0..width {
                match paragraph[end..].chars().next() {
                    Some(c) => end += c.len_utf8(),
                    None => break,
                }
            }
            if end >= paragraph.len() {
                lines.push(&paragraph[start..]);
                break;
            }
            // Prefer the last space inside the window; else cut hard at `end`.
            let cut = paragraph[start..end]
                .rfind(' ')
                .map(|pos| start + pos)
                .filter(|&pos| pos > start)
                .unwrap_or(end);
            lines.push(&paragraph[start..cut]);
            start = cut;
            // Skip the separating spaces at the next line start.
            while start < paragraph.len() && paragraph.as_bytes()[start] == b' ' {
                start += 1;
            }
        }
        lines
    }

    /// System font with a wide glyph coverage, or the built-in Helvetica
    /// fallback. The only platform-touching code in the app: a candidate OS
    /// font path list; missing files are skipped so every OS still gets a PDF.
    fn load_font(
        doc: &mut PdfDocument,
        warnings: &mut Vec<PdfWarnMsg>,
    ) -> Result<FontRef> {
        const CANDIDATES: &[&str] = &[
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ];
        for path in CANDIDATES {
            if let Some(font) = std::fs::read(path)
                .ok()
                .and_then(|bytes| ParsedFont::from_bytes(&bytes, 0, warnings))
            {
                let id = doc.add_font(&font);
                return Ok(FontRef::External(id));
            }
        }
        Ok(FontRef::Builtin(BuiltinFont::Helvetica))
    }

    #[cfg(test)]
    mod wrap_tests {
        use super::{format_pdf, wrap, CHARS_PER_LINE};
        use crate::TranscriptionSegment;

        /// Direct regression for the app crash: saving a Hebrew transcript as
        /// PDF panicked inside `format_pdf` and aborted the Tauri command with
        /// STATUS_STACK_BUFFER_OVERRUN, because the wrapper sliced a string at
        /// a non-char-boundary byte index. The export must simply succeed.
        #[test]
        fn format_pdf_hebrew_does_not_panic() {
            let segments = vec![
                TranscriptionSegment {
                    start: 0,
                    end: 3_000,
                    text: "שלום עולם, אני מדבר עברית".to_string(),
                },
                TranscriptionSegment {
                    start: 3_000,
                    end: 6_000,
                    text: "זהו טקסט ארוך מאוד שנדרש להיארך על פני מספר שורות כדי לוודא שהגלישה והסידור החזותי עובדים כמו שצריך".to_string(),
                },
            ];

            let bytes = format_pdf(&segments, true).expect("format_pdf must not panic on Hebrew");
            assert!(
                bytes.len() > 500,
                "the exported PDF must have real content"
            );
        }

        /// `wrap` is pure (no printpdf, no I/O), so it is directly testable.
        /// It must survive Hebrew: 2 bytes per char means a *byte* budget cuts
        /// mid-char, which used to panic on a non-char-boundary slice and drop
        /// the tail of the sentence. Every glyph except the separating spaces
        /// must survive, and every emitted line must be a char boundary.
        #[test]
        fn wrap_keeps_every_hebrew_glyph() {
            let text = "שלום עולם, אני מדבר עברית";
            let lines = wrap(text, CHARS_PER_LINE);

            let non_space = |s: &str| s.chars().filter(|c| !c.is_whitespace()).count();
            let joined = lines.concat();
            assert_eq!(
                non_space(&joined),
                non_space(text),
                "wrap dropped glyphs: {lines:?}"
            );
            for line in &lines {
                assert!(
                    text.contains(*line),
                    "wrap emitted a fragment not present in the source: {line:?}"
                );
            }
        }

        /// A long Hebrew sentence must split into several lines, each at most
        /// `width` characters (not bytes), and lose nothing at the seams.
        #[test]
        fn wrap_respects_char_width_and_keeps_tail() {
            let text = "שלום עולם ".repeat(20);
            let lines = wrap(&text, CHARS_PER_LINE);

            assert!(lines.len() > 1, "long text must wrap");
            for line in &lines {
                assert!(
                    line.chars().count() <= CHARS_PER_LINE,
                    "line exceeds the char budget: {line:?}"
                );
            }
            assert!(
                lines.concat().contains("עולם"),
                "the tail of the sentence must survive wrapping"
            );
        }

        /// Regression guard for the crash: mixed-width text (Hebrew + ASCII +
        /// a CJK char at 3 bytes) must never produce a non-char-boundary slice.
        #[test]
        fn wrap_never_splits_a_multibyte_char() {
            let text = format!("{} {} {} {}", "שלום", "hello", "日本語", "עולם");
            let lines = wrap(&text, 7);

            assert_eq!(
                lines,
                vec!["שלום", "hello", "日本語", "עולם"],
                "wrap must keep whole words and never split a char: {lines:?}"
            );
        }
    }
}

pub use docx::format_docx;
pub use pdf::format_pdf;

#[cfg(test)]
mod tests {
    use super::segment_line;
    use crate::format::visual_order;
    use crate::TranscriptionSegment;

    fn seg(text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start: 0,
            end: 5_000,
            text: text.to_string(),
        }
    }

    /// whisper-burn's PDF module (the exporter the app ships for Hebrew
    /// transcripts) must hand the bidi-less printpdf engine a faithful
    /// visual-order permutation of the whole sentence — like the console
    /// `bracket_line` — and must not truncate the tail.
    #[test]
    fn pdf_hebrew_keeps_full_text_in_visual_order() {
        let s = seg("שלום עולם, אני מדבר עברית");

        let logical = segment_line(&s, false);
        assert_eq!(
            logical,
            "שלום עולם, אני מדבר עברית",
            "logical order preserved"
        );

        // The mirror must be a loss-free permutation of the *whole* sentence:
        // reversing it again restores the original. This is the invariant that
        // proves no glyph was dropped on the way to the printpdf engine.
        let visual = visual_order(&logical);
        assert_ne!(
            visual, logical,
            "RTL text must be mirrored into visual order for the PDF"
        );
        assert_eq!(
            visual.chars().rev().collect::<String>(),
            logical,
            "visual order must be a faithful permutation of the whole sentence"
        );
    }
}
