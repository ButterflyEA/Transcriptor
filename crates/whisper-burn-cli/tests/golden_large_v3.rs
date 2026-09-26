//! Golden test for the exact user command that regressed: `--model large-v3`
//! with auto-detected language on the Hebrew clip. Ground truth was generated
//! from reference whisper (python) itself in docs/golden/.
//!
//! Run it explicitly - it needs the ~2.9 GB cached large-v3 checkpoint and
//! takes about 35 s on a discrete GPU:
//!
//! ```text
//! cargo test -p whisper-burn-cli --test golden_large_v3 -- --ignored
//! ```
//!
//! The clip is copied into a temp directory first: `--output json` writes its
//! sidecar next to the input, and the golden must not litter the repository.
use std::process::Command;

const EXE: &str = env!("CARGO_BIN_EXE_whisper-burn-cli");

/// Beam 5 matches the reference default for a large model, and is what the
/// golden was generated with. Greedy decoding is a different search and is
/// covered separately by the greedy/beam parity tests.
const BEAM_SIZE: &str = "5";

fn golden_path() -> std::path::PathBuf {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/golden/reference_large-v3_auto_golden.json")
        .into()
}

fn source_wav() -> std::path::PathBuf {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_16000_mono.wav").into()
}

/// Copy the clip somewhere writable and hand back (wav, json sidecar).
fn clip_in_temp_dir(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("whisper_burn_golden_{tag}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let wav = dir.join("clip.wav");
    std::fs::copy(source_wav(), &wav).expect("copy the golden clip");
    let json = wav.with_extension("json");
    (wav, json)
}

#[test]
#[ignore = "heavy: needs the ~2.9 GB cached large-v3 checkpoint (~35 s)"]
fn large_v3_auto_matches_reference_hebrew_golden() {
    let (wav, json) = clip_in_temp_dir("large_v3");

    let out = Command::new(EXE)
        .args(["--model", "large-v3", "--device", "wgpu", "--output", "json"])
        .args(["--language", "auto", "--beam-size", BEAM_SIZE])
        .arg(&wav)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // `--output json` writes a file; stdout carries human-readable bracket
    // lines, not the transcript.
    let ours: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&json).unwrap_or_else(|e| panic!("read {}: {e}", json.display())),
    )
    .expect("sidecar json parses");

    let golden = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(golden_path()).expect("golden file is present"),
    )
    .expect("golden json parses");

    // The emitted key set is part of the contract, not just the values.
    let mut our_keys: Vec<&str> = ours.as_object().unwrap().keys().map(String::as_str).collect();
    let mut golden_keys: Vec<&str> =
        golden.as_object().unwrap().keys().map(String::as_str).collect();
    our_keys.sort_unstable();
    golden_keys.sort_unstable();
    assert_eq!(our_keys, golden_keys, "top-level key set diverged");

    assert_eq!(
        ours["language"].as_str(),
        golden["language"].as_str(),
        "language detection diverged; ours={} golden={}",
        ours["language"],
        golden["language"]
    );

    let osegs = ours["segments"].as_array().expect("segments array");
    let gsegs = golden["segments"].as_array().expect("golden segments array");
    assert_eq!(
        osegs.len(),
        gsegs.len(),
        "segment count ours={} golden={}",
        osegs.len(),
        gsegs.len()
    );

    for (i, (o, g)) in osegs.iter().zip(gsegs.iter()).enumerate() {
        let mut our_keys: Vec<&str> = o.as_object().unwrap().keys().map(String::as_str).collect();
        let mut golden_keys: Vec<&str> =
            g.as_object().unwrap().keys().map(String::as_str).collect();
        our_keys.sort_unstable();
        golden_keys.sort_unstable();
        assert_eq!(our_keys, golden_keys, "segment {i} key set diverged");

        assert_eq!(o["start"], g["start"], "segment {i} start ours={} golden={}", o["start"], g["start"]);
        assert_eq!(o["end"], g["end"], "segment {i} end ours={} golden={}", o["end"], g["end"]);
        // Exact text: the reference is the ground truth, and a fuzzy match
        // would have hidden the leading-space defect this golden now guards.
        assert_eq!(o["text"], g["text"], "segment {i} text ours={:?} golden={:?}", o["text"], g["text"]);
    }

    let _ = std::fs::remove_dir_all(json.parent().unwrap());
}
