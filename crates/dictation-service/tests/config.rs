use dictation_service::config::{Config, Precision, parse_model};
use whisper_burn::ModelSize;

#[test]
fn defaults_are_valid_and_match_spec() {
    let c = Config::default();
    assert_eq!(c.hotkey, "ctrl+space");
    assert_eq!(c.model, "ivrit-turbo");
    assert_eq!(c.model_dir, None);
    assert_eq!(c.language, "he");
    assert_eq!(c.queue_cap, 3);
    assert_eq!(c.max_record_secs, 120);
    assert!(c.validate().is_ok());
    assert_eq!(c.language_option().as_deref(), Some("he"));
    let auto = Config {
        language: "auto".into(),
        ..Config::default()
    };
    assert_eq!(auto.language_option(), None);
}

#[test]
fn round_trips_through_disk() {
    let dir = std::env::temp_dir().join(format!("dict-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nested").join("config.toml");
    let c = Config {
        language: "he".into(),
        model_dir: Some(dir.join("models")),
        ..Config::default()
    };
    c.save_to(&path).unwrap();
    assert_eq!(Config::load_from(&path).unwrap(), c);
    assert_eq!(c.language_option(), Some("he".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_file_yields_default() {
    let p = std::env::temp_dir().join("dict-definitely-missing/config.toml");
    assert_eq!(Config::load_from(&p).unwrap(), Config::default());
}

#[test]
fn partial_file_fills_defaults() {
    let dir = std::env::temp_dir().join(format!("dict-cfg-partial-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    std::fs::write(&path, "language = \"he\"\n").unwrap();
    let c = Config::load_from(&path).unwrap();
    assert_eq!(c.language, "he");
    assert_eq!(c.hotkey, "ctrl+space");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_file_is_rejected() {
    let dir = std::env::temp_dir().join(format!("dict-cfg-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    std::fs::write(&path, "queue_cap = 0\n").unwrap();
    assert!(Config::load_from(&path).is_err());
    std::fs::write(&path, "not toml {{{").unwrap();
    assert!(Config::load_from(&path).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rejects_bad_values() {
    let mutations: Vec<fn(&mut Config)> = vec![
        |c| c.model = "huge".into(),
        |c| c.language = "english".into(),
        |c| c.language = "HE".into(),
        |c| c.queue_cap = 0,
        |c| c.max_record_secs = 0,
        |c| c.hotkey = "".into(),
        |c| c.stream_window_secs = 0,
        |c| c.stream_window_secs = 31,
        |c| c.stream_interval_ms = 50,
        |c| c.stream_overlap_ms = 3000,
        |c| c.stream_lock_words = 0,
    ];
    for mutate in mutations {
        let mut c = Config::default();
        mutate(&mut c);
        assert!(c.validate().is_err());
    }
}

#[test]
fn save_refuses_invalid_config() {
    let c = Config {
        queue_cap: 0,
        ..Config::default()
    };
    let p = std::env::temp_dir().join("dict-never-written/config.toml");
    assert!(c.save_to(&p).is_err());
    assert!(!p.exists());
}

#[test]
fn model_names_resolve() {
    assert_eq!(parse_model("ivrit-turbo"), Some(ModelSize::LargeV3Turbo));
    assert_eq!(parse_model("tiny"), Some(ModelSize::Tiny));
    assert_eq!(parse_model("nope"), None);
}

#[test]
fn precision_defaults_to_f16_and_parses_from_toml() {
    assert_eq!(Config::default().precision, Precision::F16);
    let c: Config = toml::from_str("precision = \"f32\"").unwrap();
    assert_eq!(c.precision, Precision::F32);
}

#[test]
fn stream_defaults_are_off_and_valid() {
    let c = Config::default();
    assert!(!c.stream);
    assert_eq!(
        (
            c.stream_window_secs,
            c.stream_interval_ms,
            c.stream_overlap_ms,
            c.stream_lock_words
        ),
        (3, 1000, 400, 2)
    );
    assert!(c.validate().is_ok());
}
