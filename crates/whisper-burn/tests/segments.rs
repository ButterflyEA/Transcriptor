use whisper_burn::decoding::segments::{assemble_segments, timestamp_token_to_ms};
use whisper_burn::tokenizer::tiktoken::CoreBpe;
use whisper_burn::tokenizer::whisper::TextTokenizer;

const MINI: &str = include_str!("fixtures/mini.tiktoken");

fn mini_tokenizer() -> TextTokenizer {
    let mut specials = vec![
        ("<|endoftext|>".to_string(), 50256),
        ("<|startoftranscript|>".to_string(), 50257),
        ("<|translate|>".to_string(), 50358),
        ("<|transcribe|>".to_string(), 50359),
        ("<|startoflm|>".to_string(), 50360),
        ("<|startofprev|>".to_string(), 50361),
        ("<|nospeech|>".to_string(), 50362),
        ("<|notimestamps|>".to_string(), 50363),
        ("<|0.00|>".to_string(), 50364),
    ];
    for (i, code) in TextTokenizer::LANGUAGES.iter().enumerate() {
        specials.push((format!("<|{code}|>"), 50258 + i as u32));
    }
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    TextTokenizer::new_from_core(core, 51865, 1500).unwrap()
}

#[test]
fn timestamp_ms_from_token() {
    let tok = mini_tokenizer();
    assert_eq!(timestamp_token_to_ms(50364, &tok), 0); // |0.00|
    assert_eq!(timestamp_token_to_ms(50365, &tok), 10); // |0.01|
    assert_eq!(timestamp_token_to_ms(50369, &tok), 50); // |0.05|
    assert_eq!(timestamp_token_to_ms(50372, &tok), 80); // |0.08|
}

#[test]
fn segments_split_on_timestamps() {
    let tok = mini_tokenizer();
    // <|0.00|> hello <|0.05|> hello! <|0.08|> <|eot|>
    let sequence = vec![50364, 3, 50369, 4, 50372, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 50);
    assert_eq!(segs[0].text, "hello");
    assert_eq!(segs[1].start, 50);
    assert_eq!(segs[1].end, 80);
    assert_eq!(segs[1].text, "hello!");
}

#[test]
fn seek_offsets_all_times() {
    let tok = mini_tokenizer();
    let sequence = vec![50364, 3, 50369, 4, 50372, 50256];
    let segs = assemble_segments(&sequence, &tok, 61000).unwrap();
    assert_eq!(segs[0].start, 61000);
    assert_eq!(segs[0].end, 61050);
    assert_eq!(segs[1].start, 61050);
    assert_eq!(segs[1].end, 61080);
}

#[test]
fn empty_runs_are_skipped_but_advance_time() {
    let tok = mini_tokenizer();
    // <|0.00|> hi <|0.01|> <|0.03|> hello! <|0.05|> <|eot|>
    let sequence = vec![50364, 2, 50365, 50367, 4, 50369, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 10);
    assert_eq!(segs[0].text, "hi");
    assert_eq!(segs[1].start, 30);
    assert_eq!(segs[1].end, 50);
    assert_eq!(segs[1].text, "hello!");
}

#[test]
fn trailing_text_without_closing_timestamp_is_dropped() {
    let tok = mini_tokenizer();
    // <|0.00|> hello <|0.05|> hello! <|eot|>  (no timestamp after "hello!")
    let sequence = vec![50364, 3, 50369, 4, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 50);
    assert_eq!(segs[0].text, "hello");
}

#[test]
fn ignores_sot_sequence_prefix_and_leading_text() {
    let tok = mini_tokenizer();
    // <|sot|> <|en|> <|transcribe|> lead <|0.00|> hello <|0.05|> <|eot|>
    let sequence = vec![50257, 50258, 50359, 2, 50364, 3, 50369, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 50);
    assert_eq!(segs[0].text, "hello");
}
