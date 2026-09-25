//! Golden test for the exact user command that regressed: `--model large-v3`
//! auto-detected language on the Hebrew clip. Ground truth was generated from
//! reference whisper (python) itself in docs/golden/.
use std::process::Command;

const EXE: &str = env!("CARGO_BIN_EXE_whisper-burn-cli");

#[test]
#[ignore = "network + heavy: needs cached large-v3 weights"]
fn large_v3_auto_matches_reference_hebrew_golden() {
    let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_16000_mono.wav");
    let out = Command::new(EXE)
        .args(["--model", "large-v3", "--device", "cpu", "--output", "json"])
        .arg(wav)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ours: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("json transcript on stdout");

    let golden = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/golden/reference_large-v3_auto_golden.json"
        ))
        .unwrap(),
    )
    .unwrap();

    assert_eq!(
        ours["language"].as_str(),
        golden["language"].as_str(),
        "language detection diverged; ours={} golden={}",
        ours["language"],
        golden["language"]
    );

    let osegs = ours["segments"].as_array().unwrap();
    let gsegs = golden["segments"].as_array().unwrap();
    assert_eq!(osegs.len(), gsegs.len(), "segment count ours={} golden={}", osegs.len(), gsegs.len());
    for (o, g) in osegs.iter().zip(gsegs.iter()) {
        assert_eq!(o["start"], g["start"], "start ours={} golden={}", o["start"], g["start"]);
        assert_eq!(o["end"], g["end"], "end ours={} golden={}", o["end"], g["end"]);
        assert_eq!(
            o["text"],
            g["text"],
            "text ours={:?} golden={:?}",
            o["text"],
            g["text"]
        );
    }
}
