#[cfg(feature = "weights")]
use std::sync::{Mutex, MutexGuard, OnceLock};
use whisper_burn::config::ModelSize;
use whisper_burn::host::{HostCapacity, detect, fits, recommend};
#[cfg(feature = "weights")]
use whisper_burn::host::resident_bytes;

/// `WHISPER_BURN_CACHE` is process-global, so the tests that move it must not
/// run concurrently or they will read each other's directory.
#[cfg(feature = "weights")]
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const GIB: u64 = 1024 * 1024 * 1024;

fn gib(n: u64) -> u64 {
    n * GIB
}

fn capacity(available_gib: u64) -> HostCapacity {
    HostCapacity { total_bytes: gib(available_gib + 8), available_bytes: gib(available_gib) }
}

#[test]
fn estimate_must_leave_thirty_percent_headroom() {
    // Exactly at 70% of available is the last accepted size.
    assert!(fits(gib(7), gib(10)));
    assert!(!fits(gib(7) + 1, gib(10)));
}

#[test]
fn nothing_fits_a_zero_capacity_host() {
    assert!(!fits(1, 0));
}

#[test]
fn eight_gib_vm_recommends_turbo_but_never_large_v3() {
    // The machine from the crash: 8 GB total, 5 GB free once the desktop has
    // its share. Budget is 70% of 5 GiB = 3584 MiB, which admits turbo
    // (3300 MiB) and stops well short of the 6200 MiB models.
    let cap = capacity(5);
    assert!(!fits(ModelSize::LargeV3.approx_resident_bytes(), cap.available_bytes));
    assert_eq!(recommend(&cap, &ModelSize::ALL), Some(ModelSize::LargeV3Turbo));
}

#[test]
fn roomy_host_recommends_the_largest_model() {
    let cap = capacity(256);
    let got = recommend(&cap, &ModelSize::ALL).expect("a 256 GB host fits something");
    assert_eq!(got.approx_resident_bytes(), ModelSize::LargeV3.approx_resident_bytes());
}

#[test]
fn recommendation_never_exceeds_available_when_a_smaller_model_also_fits() {
    let cap = capacity(6);
    for model in ModelSize::ALL {
        if !fits(model.approx_resident_bytes(), cap.available_bytes) {
            continue;
        }
        let got = recommend(&cap, &ModelSize::ALL).expect("a fitting model exists");
        assert!(
            got.approx_resident_bytes() >= model.approx_resident_bytes(),
            "recommended {got:?} is smaller than fitting {model:?}"
        );
    }
}

#[test]
fn a_one_gib_host_recommends_only_the_smallest_models() {
    // Budget is 70% of 1 GiB = 717 MiB. tiny (155) and base (290) fit;
    // small (976) does not.
    let cap = capacity(1);
    assert!(!fits(ModelSize::Small.approx_resident_bytes(), cap.available_bytes));
    assert_eq!(recommend(&cap, &ModelSize::ALL), Some(ModelSize::Base));
}

#[test]
fn an_empty_model_list_recommends_nothing() {
    assert_eq!(recommend(&capacity(64), &[]), None);
}

#[test]
fn a_tiny_capacity_recommends_nothing_at_all() {
    assert_eq!(recommend(&capacity(0), &ModelSize::ALL), None);
}

#[test]
fn detect_reports_a_plausible_machine() {
    let cap = detect().expect("a real host must report memory");
    assert!(cap.total_bytes > 0);
    assert!(cap.available_bytes <= cap.total_bytes, "{cap:?}");
    assert!(cap.available_bytes > 0, "{cap:?}");
}

#[cfg(feature = "weights")]
#[test]
fn resident_bytes_falls_back_to_the_table_without_a_checkpoint() {
    // Point the cache at an empty directory: this machine has real checkpoints
    // in ~/.cache/whisper-burn, and the fallback must not depend on that.
    let dir = std::env::temp_dir().join(format!("whisper-burn-host-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp cache");

    let _guard = env_lock();
    // SAFETY: the guard above serialises every test that touches this var.
    unsafe { std::env::set_var("WHISPER_BURN_CACHE", &dir) };
    let got = resident_bytes(ModelSize::Small);
    unsafe { std::env::remove_var("WHISPER_BURN_CACHE") };
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(got, ModelSize::Small.approx_resident_bytes());
}

#[cfg(feature = "weights")]
#[test]
fn resident_bytes_prefers_the_real_checkpoint_when_present() {
    let dir = std::env::temp_dir().join(format!("whisper-burn-host-{}", std::process::id()));
    let repo = dir.join(ModelSize::Small.repo_id());
    std::fs::create_dir_all(&repo).expect("create temp cache");
    let weights = repo.join("model.safetensors");
    std::fs::write(&weights, vec![0u8; 4096]).expect("write fake checkpoint");

    let _guard = env_lock();
    // SAFETY: the guard above serialises every test that touches this var.
    unsafe { std::env::set_var("WHISPER_BURN_CACHE", &dir) };
    let got = resident_bytes(ModelSize::Small);
    unsafe { std::env::remove_var("WHISPER_BURN_CACHE") };
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(got, 4096, "the real file size must win over the table");
}

#[cfg(feature = "weights")]
#[test]
fn resident_bytes_ignores_a_zero_length_checkpoint() {
    let dir = std::env::temp_dir().join(format!("whisper-burn-host-zero-{}", std::process::id()));
    let repo = dir.join(ModelSize::Small.repo_id());
    std::fs::create_dir_all(&repo).expect("create temp cache");
    std::fs::write(repo.join("model.safetensors"), b"").expect("write empty file");

    let _guard = env_lock();
    // SAFETY: the guard above serialises every test that touches this var.
    unsafe { std::env::set_var("WHISPER_BURN_CACHE", &dir) };
    let got = resident_bytes(ModelSize::Small);
    unsafe { std::env::remove_var("WHISPER_BURN_CACHE") };
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(got, ModelSize::Small.approx_resident_bytes());
}
