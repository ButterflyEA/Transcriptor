use std::path::{Path, PathBuf};

use dictation_service::engine::{join_segments, resolve_model_dir};
use whisper_burn::TranscriptionSegment;

fn seg(t: &str) -> TranscriptionSegment {
    TranscriptionSegment {
        start: 0,
        end: 1000,
        text: t.into(),
    }
}

#[test]
fn joins_trims_and_collapses_whitespace() {
    let s = [seg(" Hello"), seg("  world. "), seg("")];
    assert_eq!(join_segments(&s), "Hello world.");
}

#[test]
fn empty_is_empty() {
    assert_eq!(join_segments(&[]), "");
}

#[test]
fn hebrew_text_is_preserved() {
    let s = [seg(" שלום"), seg(" עולם")];
    assert_eq!(join_segments(&s), "שלום עולם");
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("dict-eng-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }

    fn model(&self, rel: &str) -> PathBuf {
        let d = self.0.join(rel);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("model.safetensors"), b"x").unwrap();
        d
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn configured_dir_wins() {
    let t = Tmp::new("cfg");
    let configured = t.model("custom");
    let exe = t.0.join("exe");
    t.model("exe/models/ivrit-turbo");
    let got = resolve_model_dir(Some(&configured), &exe, &t.0.join("data"));
    assert_eq!(got, Some(configured));
}

#[test]
fn falls_back_to_exe_then_data_dir() {
    let t = Tmp::new("fallback");
    let exe = t.0.join("exe");
    let data = t.0.join("data");
    let data_model = t.model("data/models/ivrit-turbo");
    assert_eq!(resolve_model_dir(None, &exe, &data), Some(data_model));
    let exe_model = t.model("exe/models/ivrit-turbo");
    assert_eq!(resolve_model_dir(None, &exe, &data), Some(exe_model));
}

#[test]
fn configured_dir_without_model_is_skipped() {
    let t = Tmp::new("skip");
    let empty = t.0.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let exe_model = t.model("exe/models/ivrit-turbo");
    let got = resolve_model_dir(Some(&empty), &t.0.join("exe"), &t.0.join("data"));
    assert_eq!(got, Some(exe_model));
}

#[test]
fn missing_everywhere_is_none() {
    let t = Tmp::new("none");
    assert_eq!(
        resolve_model_dir(None, &t.0.join("exe"), &t.0.join("data")),
        None
    );
}

#[test]
fn loading_a_missing_model_errors_without_network() {
    let missing = Path::new("definitely-not-a-real-model-dir");
    let err =
        dictation_service::engine::Engine::load(whisper_burn::ModelSize::LargeV3Turbo, missing);
    assert!(err.is_err());
}

#[test]
#[ignore = "needs the model files in DICTATION_MODEL_DIR"]
fn transcribes_silence_without_hallucinating() {
    let dir = PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let engine =
        dictation_service::engine::Engine::load(whisper_burn::ModelSize::LargeV3Turbo, &dir)
            .unwrap();
    let pcm = vec![0.0f32; 16_000 * 2];
    let text = engine.transcribe_pcm(&pcm, 16_000, Some("he")).unwrap();
    assert!(text.len() < 40, "silence hallucinated: {text:?}");
}
