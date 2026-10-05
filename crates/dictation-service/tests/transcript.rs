use dictation_service::transcript::*;

fn v(s: &str) -> Vec<String> {
    words(s)
}

#[test]
fn normalise_strips_punctuation_and_niqqud() {
    assert_eq!(normalize_word("שָׁלוֹם,"), normalize_word("שלום"));
    assert_eq!(normalize_word("Hello."), "hello");
}

#[test]
fn agreed_prefix_ignores_punctuation_flips() {
    assert_eq!(
        agreed_prefix_len(&v("שלום, זאת הקלטה"), &v("שלום. זאת הקלטת")),
        2
    );
}

#[test]
fn extending_pass_appends_only() {
    let mut t = Transcript::new(2);
    assert_eq!(t.on_pass("שלום"), Some(Edit::Append("שלום".into())));
    assert_eq!(t.on_pass("שלום עולם"), Some(Edit::Append(" עולם".into())));
}

#[test]
fn changed_tail_is_replaced_with_exact_backspace_count() {
    let mut t = Transcript::new(2);
    t.on_pass("אני גר בגבעת אב");
    assert_eq!(
        t.on_pass("אני גר בגבעת אבני"),
        Some(Edit::Append("ני".into()))
    );
    assert_eq!(
        t.on_pass("בגבעת רפאל"),
        Some(Edit::Replace {
            backspaces: 4,
            text: "רפאל".into()
        })
    );
}

#[test]
fn identical_pass_emits_nothing() {
    let mut t = Transcript::new(2);
    t.on_pass("אחד שניים");
    assert_eq!(t.on_pass("אחד שניים"), None);
}

#[test]
fn agreed_old_words_lock_and_later_passes_cover_only_new_audio() {
    let mut t = Transcript::new(2);
    t.on_pass("אחד שניים שלושה ארבעה");
    t.on_pass("אחד שניים שלושה ארבעה");
    assert_eq!(t.locked_words(), 2);
    assert_eq!(
        t.on_pass("שלושה ארבעה חמישה"),
        Some(Edit::Append(" חמישה".into()))
    );
}

#[test]
fn append_only_never_backspaces() {
    let mut t = Transcript::new(2);
    t.on_pass("אחד שניים");
    t.set_append_only();
    let e = t.on_pass("אחד שלושה ארבעה");
    assert_eq!(e, Some(Edit::Append(" ארבעה".into())));
}

#[test]
fn finish_corrects_the_tail_and_returns_full_text() {
    let mut t = Transcript::new(1);
    t.on_pass("אחד שניים");
    t.on_pass("אחד שניים");
    let (edit, full) = t.finish("שניים שלושה");
    assert_eq!(edit, Some(Edit::Append(" שלושה".into())));
    assert_eq!(full, "אחד שניים שלושה");
}
