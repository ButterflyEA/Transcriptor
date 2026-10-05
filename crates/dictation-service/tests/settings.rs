use dictation_service::config::Config;
use dictation_service::settings::{FormState, chord_string, validate_form};

fn form() -> FormState {
    FormState::from_config(&Config::default())
}

#[test]
fn default_form_is_valid() {
    assert_eq!(validate_form(&form()).unwrap(), Config::default());
}

#[test]
fn edits_are_applied_and_other_settings_kept() {
    let base = Config {
        paste_delay_ms: 123,
        ..Config::default()
    };
    let mut f = FormState::from_config(&base);
    f.language = " HE ".into();
    f.model_dir = r"C:\models\x".into();
    f.input_device = "Mic".into();
    let c = validate_form(&f).unwrap();
    assert_eq!(c.language, "he");
    assert_eq!(
        c.model_dir.as_deref(),
        Some(std::path::Path::new(r"C:\models\x"))
    );
    assert_eq!(c.input_device.as_deref(), Some("Mic"));
    assert_eq!(c.paste_delay_ms, 123);
}

#[test]
fn empty_model_dir_and_device_mean_none() {
    let c = validate_form(&form()).unwrap();
    assert!(c.model_dir.is_none() && c.input_device.is_none());
}

#[test]
fn rejects_bad_language_model_and_hotkey() {
    let mut f = form();
    f.language = "hebrew".into();
    assert!(validate_form(&f).is_err());
    let mut f = form();
    f.model = "nope".into();
    assert!(validate_form(&f).is_err());
    let mut f = form();
    f.hotkey = " + ".into();
    assert!(validate_form(&f).is_err());
}

#[test]
fn recorded_chord_round_trips_through_the_parser() {
    let s = chord_string(true, true, false, false, "space");
    assert_eq!(s, "ctrl+shift+space");
    let mut f = form();
    f.hotkey = s;
    assert!(validate_form(&f).is_ok());
}
