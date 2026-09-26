use whisper_burn::decoding::segments::{
    TIME_PRECISION_MS, assemble_segments, timestamp_token_to_ms,
};
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
fn timestamp_steps_are_twenty_milliseconds() {
    let tok = mini_tokenizer();
    // The reference names its timestamp specials `<|{i * 0.02:.2f}|>`, so one
    // step is 20 ms: 50 steps is one second, 1500 steps the 30 s window.
    assert_eq!(TIME_PRECISION_MS, 20);
    assert_eq!(timestamp_token_to_ms(50364, &tok), 0); // <|0.00|>
    assert_eq!(timestamp_token_to_ms(50365, &tok), 20); // <|0.02|>
    assert_eq!(timestamp_token_to_ms(50414, &tok), 1_000); // <|1.00|>
    assert_eq!(timestamp_token_to_ms(50364 + 1500, &tok), 30_000); // <|30.00|>
}

#[test]
fn timestamp_ms_from_token() {
    let tok = mini_tokenizer();
    assert_eq!(timestamp_token_to_ms(50364, &tok), 0); // |0.00|
    assert_eq!(timestamp_token_to_ms(50365, &tok), 20); // |0.02|
    assert_eq!(timestamp_token_to_ms(50369, &tok), 100); // |0.10|
    assert_eq!(timestamp_token_to_ms(50372, &tok), 160); // |0.16|
}

#[test]
fn segments_split_on_timestamps() {
    let tok = mini_tokenizer();
    // <|0.00|> hello <|0.10|> hello! <|0.16|> <|eot|>
    let sequence = vec![50364, 3, 50369, 4, 50372, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 100);
    assert_eq!(segs[0].text, "hello");
    assert_eq!(segs[1].start, 100);
    assert_eq!(segs[1].end, 160);
    assert_eq!(segs[1].text, "hello!");
}

/// Whisper marks a word start with the BPE space marker, so the first text
/// token after `<|ts|>` decodes to " hello". The timestamp already supplies
/// the word break, so segment text starts flush against the word - the
/// reference golden has no leading space, and one costs a byte-level
/// mismatch on every segment.
#[test]
fn segment_text_drops_the_post_timestamp_space() {
    let tok = mini_tokenizer();
    // <|0.00|> <space> hello <|0.10|> <|eot|>: the BPE word-start marker in
    // front of the first real token is what a real decoded window looks like.
    // Token 3 is "hello" in this fixture.
    let space = tok.encode(" ").expect("fixture tokenises a bare space");
    let mut sequence = vec![50364];
    sequence.extend(space.iter().copied());
    sequence.push(3);
    sequence.extend([50369, 50256]);

    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs[0].text, "hello", "leading space must be dropped");
}

#[test]
fn seek_offsets_all_times() {
    let tok = mini_tokenizer();
    let sequence = vec![50364, 3, 50369, 4, 50372, 50256];
    let segs = assemble_segments(&sequence, &tok, 61000).unwrap();
    assert_eq!(segs[0].start, 61000);
    assert_eq!(segs[0].end, 61100);
    assert_eq!(segs[1].start, 61100);
    assert_eq!(segs[1].end, 61160);
}

#[test]
fn empty_runs_are_skipped_but_advance_time() {
    let tok = mini_tokenizer();
    // <|0.00|> hi <|0.02|> <|0.06|> hello! <|0.10|> <|eot|>
    let sequence = vec![50364, 2, 50365, 50367, 4, 50369, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 20);
    assert_eq!(segs[0].text, "hi");
    assert_eq!(segs[1].start, 60);
    assert_eq!(segs[1].end, 100);
    assert_eq!(segs[1].text, "hello!");
}

#[test]
fn trailing_text_without_closing_timestamp_is_dropped() {
    let tok = mini_tokenizer();
    // <|0.00|> hello <|0.10|> hello! <|eot|>  (no timestamp after "hello!")
    let sequence = vec![50364, 3, 50369, 4, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 100);
    assert_eq!(segs[0].text, "hello");
}

#[test]
fn ignores_sot_sequence_prefix_and_leading_text() {
    let tok = mini_tokenizer();
    // <|sot|> <|en|> <|transcribe|> lead <|0.00|> hello <|0.10|> <|eot|>
    let sequence = vec![50257, 50258, 50359, 2, 50364, 3, 50369, 50256];
    let segs = assemble_segments(&sequence, &tok, 0).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 100);
    assert_eq!(segs[0].text, "hello");
}
