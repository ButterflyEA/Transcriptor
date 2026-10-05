use dictation_service::metrics::{char_error_rate, edit_distance, normalize_for_compare};

#[test]
fn edit_distance_counts_chars() {
    assert_eq!(edit_distance("שלום", "שלום"), 0);
    assert_eq!(edit_distance("שלום", "שלם"), 1);
    assert_eq!(edit_distance("", "אבג"), 3);
    assert_eq!(edit_distance("kitten", "sitting"), 3);
}

#[test]
fn normalisation_ignores_punctuation_niqqud_and_spacing() {
    assert_eq!(normalize_for_compare("שָׁלוֹם,  עולם."), "שלום עולם");
    assert_eq!(normalize_for_compare("Hello,  World"), "hello world");
}

#[test]
fn cer_is_relative_to_the_reference() {
    assert_eq!(char_error_rate("אבגד", "אבגד"), 0.0);
    assert!((char_error_rate("אבגד", "אבג") - 0.25).abs() < 1e-9);
    assert_eq!(char_error_rate("", ""), 0.0);
}