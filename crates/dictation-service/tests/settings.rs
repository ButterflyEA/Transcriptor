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
    let dir = std::env::temp_dir().join("dictation-settings-edit");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
    f.model_dir = dir.display().to_string();
    f.input_device = "Mic".into();
    let c = validate_form(&f).unwrap();
    assert_eq!(c.language, "he");
    assert_eq!(c.model_dir.as_deref(), Some(dir.as_path()));
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

#[test]
fn model_without_weights_is_rejected_with_a_clear_message() {
    let mut f = form();
    f.model = "tiny".into();
    let err = validate_form(&f).unwrap_err();
    assert!(err.contains("model.safetensors"), "{err}");

    let mut f = form();
    f.model_dir = std::env::temp_dir()
        .join("dictation-empty-model-dir")
        .display()
        .to_string();
    assert!(validate_form(&f).is_err());
}

#[test]
fn only_models_with_weights_are_offered() {
    let root = std::env::temp_dir().join("dictation-models-list");
    let _ = std::fs::remove_dir_all(&root);
    let tiny = root.join("models").join("tiny");
    std::fs::create_dir_all(&tiny).unwrap();
    std::fs::write(tiny.join("model.safetensors"), b"x").unwrap();
    std::fs::create_dir_all(root.join("models").join("base")).unwrap();
    std::fs::create_dir_all(root.join("models").join("not-a-model")).unwrap();
    let got = dictation_service::settings::available_models(&root, &root.join("none"));
    assert_eq!(got, vec!["ivrit-turbo".to_string(), "tiny".to_string()]);
}
