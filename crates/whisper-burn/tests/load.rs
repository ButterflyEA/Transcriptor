mod common;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "weights")]
use std::sync::Mutex;
use whisper_burn::Error;
use whisper_burn::config::ModelSize;
use whisper_burn::model::whisper::Whisper;

type B = NdArray<f32>;

#[cfg(feature = "weights")]
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("wburn_load_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn load_from_fake_checkpoint_succeeds_and_drains() {
    let tmp = tmp_dir("ok");
    common::write_fake_checkpoint(ModelSize::Tiny, &tmp);
    let w = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdArrayDevice::default()).unwrap();
    let e = ModelSize::Tiny.expected_dims();
    assert_eq!(w.dims.n_vocab, e.n_vocab);
    assert_eq!(w.dims.n_mels, e.n_mels);
    assert_eq!(w.dims.n_audio_ctx, 1500);
    assert_eq!(w.dims.n_text_ctx, 448);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn load_missing_weights_is_typed_error() {
    let tmp = tmp_dir("missing");
    let e = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdArrayDevice::default()).unwrap_err();
    assert!(matches!(e, Error::MissingWeights { .. }), "got {e:?}");
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn missing_single_weight_reports_key() {
    let tmp = tmp_dir("rmkey");
    common::write_fake_checkpoint_without(ModelSize::Tiny, &tmp, "encoder.ln_post.bias");
    let e = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdArrayDevice::default()).unwrap_err();
    assert!(
        matches!(&e, Error::MissingWeight(name) if name.contains("encoder.ln_post.bias")),
        "got {e:?}"
    );
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn config_mismatch_with_size_is_rejected() {
    let tmp = tmp_dir("cfg");
    common::write_fake_checkpoint(ModelSize::Tiny, &tmp);
    let cfg = std::fs::read_to_string(tmp.join("config.json")).unwrap();
    std::fs::write(
        tmp.join("config.json"),
        cfg.replace("\"num_mel_bins\": 80", "\"num_mel_bins\": 64"),
    )
    .unwrap();
    let e = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdArrayDevice::default()).unwrap_err();
    assert!(matches!(e, Error::ConfigParse(_)), "got {e:?}");
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[cfg(feature = "weights")]
#[test]
fn from_pretrained_loads_from_cache() {
    let _guard = ENV_LOCK.lock().unwrap();
    let cache = tmp_dir("pt_cache");
    let tgt = cache.join("openai").join("whisper-tiny");
    common::write_fake_checkpoint(ModelSize::Tiny, &tgt);
    let prev = std::env::var_os("WHISPER_BURN_CACHE");
    unsafe {
        std::env::set_var("WHISPER_BURN_CACHE", &cache);
    }
    let w = Whisper::<B>::from_pretrained(ModelSize::Tiny, NdArrayDevice::default()).unwrap();
    assert_eq!(w.dims.n_vocab, 51865);
    match prev {
        Some(v) => unsafe { std::env::set_var("WHISPER_BURN_CACHE", v) },
        None => unsafe { std::env::remove_var("WHISPER_BURN_CACHE") },
    }
    std::fs::remove_dir_all(&cache).unwrap();
}

#[cfg(not(feature = "weights"))]
#[test]
fn from_pretrained_errors_when_weights_disabled() {
    let e = Whisper::<B>::from_pretrained(ModelSize::Tiny, NdArrayDevice::default()).unwrap_err();
    assert!(matches!(e, Error::MissingWeights { .. }), "got {e:?}");
}
