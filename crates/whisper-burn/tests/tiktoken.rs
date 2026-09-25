use whisper_burn::tokenizer::tiktoken::CoreBpe;

const MINI: &str = include_str!("fixtures/mini.tiktoken");

#[test]
fn parses_and_decodes() {
    let bpe = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    assert_eq!(bpe.decode(&[3]).unwrap(), "hello");
    assert_eq!(bpe.decode(&[4]).unwrap(), "hello!");
    assert_eq!(bpe.n_vocab(), 8);
}

#[test]
fn encode_uses_bpe_merges() {
    let bpe = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    assert_eq!(bpe.encode_ordinary("hel").unwrap(), vec![6]);
    assert_eq!(bpe.encode_ordinary("low").unwrap(), vec![7]);
    let ids = bpe.encode_ordinary("hello").unwrap();
    assert!(ids.len() <= 2);
    assert_eq!(bpe.decode(&ids).unwrap(), "hello");
}

#[test]
fn special_tokens_are_numbered() {
    let bpe = CoreBpe::from_tiktoken_with_special(
        MINI.as_bytes(),
        &[("<|sot|>".into(), 8), ("<|eot|>".into(), 9)],
    )
    .unwrap();
    assert_eq!(bpe.special("<|sot|>"), Some(8));
    assert!(bpe.special("<|nope|>").is_none());
}
