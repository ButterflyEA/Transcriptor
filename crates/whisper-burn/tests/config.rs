use whisper_burn::config::{Dims, ModelDimensions, ModelSize};

const TINY: &str = include_str!("fixtures/tiny_config.json");

#[test]
fn parses_tiny_config() {
    let d = ModelDimensions::from_config_json(TINY).unwrap();
    assert_eq!(d.n_mels, 80);
    assert_eq!(d.n_audio_layer, 4);
    assert_eq!(d.n_text_layer, 4);
    assert_eq!(d.n_audio_state, 384);
    assert_eq!(d.n_text_state, 384);
    assert_eq!(d.n_head, 6);
    assert_eq!(d.n_vocab, 51865);
    assert_eq!(d.n_audio_ctx, 1500);
    assert_eq!(d.n_text_ctx, 448);
}

#[test]
fn missing_key_errors() {
    let bad = TINY.replace("\"num_mel_bins\": 80", "\"zzz\": 80");
    assert!(ModelDimensions::from_config_json(&bad).is_err());
}

#[test]
fn repo_ids_and_expected_dims() {
    assert_eq!(ModelSize::Tiny.repo_id(), "openai/whisper-tiny");
    assert_eq!(ModelSize::LargeV3.repo_id(), "openai/whisper-large-v3");
    assert_eq!(ModelSize::IvritHebrew.repo_id(), "ivrit-ai/whisper-large-v3");
    assert_eq!(
        ModelSize::LargeV3Turbo.repo_id(),
        "openai/whisper-large-v3-turbo"
    );
    assert_eq!(
        ModelSize::IvritHebrew.expected_dims(),
        ModelSize::LargeV3.expected_dims()
    );
    assert_eq!(
        ModelSize::LargeV3.expected_dims(),
        Dims {
            n_mels: 128,
            n_vocab: 51866,
            n_audio_layer: 32,
            n_text_layer: 32
        }
    );
    assert_eq!(
        ModelSize::LargeV3Turbo.expected_dims(),
        Dims {
            n_mels: 128,
            n_vocab: 51866,
            n_audio_layer: 32,
            n_text_layer: 4
        }
    );
}

#[test]
fn validate_rejects_wrong_mel_bins() {
    let d = ModelDimensions::from_config_json(TINY).unwrap();
    assert!(ModelSize::Small.validate(&d).is_err());
    assert!(ModelSize::Tiny.validate(&d).is_ok());
}

#[test]
fn parse_and_cli_name_round_trip_for_every_model() {
    for model in ModelSize::ALL {
        assert_eq!(ModelSize::parse(model.cli_name()).unwrap(), model);
    }
    assert_eq!(ModelSize::parse("ivrit-hebrew").unwrap(), ModelSize::IvritHebrew);
    assert!(ModelSize::parse("nope").is_err());
}
