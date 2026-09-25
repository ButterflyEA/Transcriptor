use whisper_burn::tokenizer::tiktoken::CoreBpe;
use whisper_burn::tokenizer::whisper::TextTokenizer;

const MINI: &str = include_str!("fixtures/mini.tiktoken");

#[test]
fn special_ids_are_fixed() {
    let mut specials = vec![
        ("<|endoftext|>".to_string(), 50257),
        ("<|startoftranscript|>".to_string(), 50258),
        ("<|translate|>".to_string(), 50358),
        ("<|transcribe|>".to_string(), 50359),
        ("<|startoflm|>".to_string(), 50360),
        ("<|startofprev|>".to_string(), 50361),
        ("<|nospeech|>".to_string(), 50362),
        ("<|notimestamps|>".to_string(), 50363),
    ];
    for (i, code) in TextTokenizer::LANGUAGES.iter().enumerate() {
        specials.push((format!("<|{code}|>"), 50259 + i as u32));
    }
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    let t = TextTokenizer::new_from_core(core, 51865, 1500).unwrap();

    assert_eq!(t.sot(), 50258);
    assert_eq!(t.eot(), 50257);
    assert_eq!(t.translate(), 50358);
    assert_eq!(t.transcribe(), 50359);
    assert_eq!(t.nospeech(), 50362);
    assert_eq!(t.notimestamps(), 50363);
    assert_eq!(t.timestamp_begin, 50364);
    assert_eq!(t.language_token("en"), Some(50259));
    assert_eq!(t.language_token("zh"), Some(50260));
    assert_eq!(t.timestamp_token(0.0), 50364);
    assert_eq!(t.timestamp_token(30.0), 51864);
}

#[test]
fn large_v3_special_tokens_shift() {
    let specials = vec![
        ("<|endoftext|>".to_string(), 50257),
        ("<|startoftranscript|>".to_string(), 50258),
        ("<|yue|>".to_string(), 50358),
        ("<|translate|>".to_string(), 50359),
        ("<|transcribe|>".to_string(), 50360),
        ("<|startoflm|>".to_string(), 50361),
        ("<|startofprev|>".to_string(), 50362),
        ("<|nospeech|>".to_string(), 50363),
        ("<|notimestamps|>".to_string(), 50364),
    ];
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    let t = TextTokenizer::new_from_core(core, 51866, 1500).unwrap();
    assert_eq!(t.language_token("en"), Some(50259));
    // <|yue|> is the 100th language token (50358), so the task tokens and
    // timestamps shift up by one vs the v2 layout (50364 -> notimestamps is
    // no longer the first timestamp; 0.00 now sits at 50365)
    assert_eq!(t.language_token("yue"), Some(50358));
    assert_eq!(t.translate(), 50359);
    assert_eq!(t.transcribe(), 50360);
    assert_eq!(t.nospeech(), 50363);
    assert_eq!(t.notimestamps(), 50364);
    assert_eq!(t.timestamp_begin, 50365);
    assert_eq!(t.timestamp_token(0.0), 50365);
    assert_eq!(t.timestamp_token(30.0), 51865);
}
