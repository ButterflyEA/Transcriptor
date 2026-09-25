//! CLI black-box tests: spawn the compiled binary and check its behavior.

use std::process::Command;

const EXE: &str = env!("CARGO_BIN_EXE_whisper-burn-cli");

#[test]
fn cli_help_lists_flags() {
    let out = Command::new(EXE).arg("--help").output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--model",
        "--language",
        "--task",
        "--beam-size",
        "--verbose",
        "--output",
        "--device",
        "--progress",
        // positional audio argument
        "<AUDIO>",
    ] {
        assert!(s.contains(flag), "help missing {flag}:\n{s}");
    }
}

#[test]
fn cli_rejects_missing_audio_arg() {
    let out = Command::new(EXE).output().unwrap();
    assert!(
        !out.status.success(),
        "expected failure, got: {}",
        out.status
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("AUDIO"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_reports_errors_for_missing_file() {
    let out = Command::new(EXE)
        .arg("does-not-exist.wav")
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "expected failure, got: {}",
        out.status
    );
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .to_lowercase()
            .contains("error"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn turbo_translate_fails_fast_with_guidance() {
    // whisper-large-v3-turbo was fine-tuned on transcription data only and
    // cannot translate (see OpenAI model card); reject the combination up
    // front instead of silently re-transcribing the source language.
    let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_16000_mono.wav");
    let out = Command::new(EXE)
        .args([
            "--model",
            "large-v3-turbo",
            "--task",
            "translate",
            "--device",
            "cpu",
        ])
        .arg(wav)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "expected failure, got: {}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not trained for translation"),
        "stderr missing turbo guidance:\n{stderr}"
    );
    assert!(
        stderr.contains("error:"),
        "expected an error line:\n{stderr}"
    );
}

#[test]
fn cli_accepts_ivrit_hebrew_model() {
    let out = Command::new(EXE)
        .args(["--model", "ivrit-hebrew"])
        .arg("does-not-exist.wav")
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "expected failure, got: {}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("unknown model"),
        "parse rejected ivrit-hebrew:\n{stderr}"
    );
    assert!(stderr.contains("error"), "stderr:\n{stderr}");
}

#[test]
#[ignore = "network"]
fn cli_prints_a_transcription() {
    let dir = std::env::temp_dir().join("whisper-burn-cli-t26");
    std::fs::create_dir_all(&dir).unwrap();
    let flac = whisper_burn::download::download_to(
        "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        &dir.join("jfk.flac"),
        false,
    )
    .unwrap();
    // copy so the transcript files land in an isolated output dir
    let out_dir = dir.join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let audio = out_dir.join("jfk.flac");
    std::fs::copy(&flac, &audio).unwrap();

    let out = Command::new(EXE)
        .args(["--language", "en", "--device", "cpu", "--output", "srt"])
        .arg(&audio)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("-->"),
        "no timestamp arrow on stdout:\n{stdout}"
    );
    assert!(
        stdout.to_lowercase().contains("country"),
        "expected the JFK quote, got:\n{stdout}"
    );

    let srt_path = out_dir.join("jfk.srt");
    assert!(
        srt_path.exists(),
        "expected {} to be written",
        srt_path.display()
    );
    let srt = std::fs::read_to_string(&srt_path).unwrap();
    assert!(srt.contains("-->"), "srt missing cues:\n{srt}");
    assert!(srt.to_lowercase().contains("country"), "srt text:\n{srt}");
}

#[test]
#[ignore = "network"]
fn cli_progress_streams_stages_and_segments() {
    let dir = std::env::temp_dir().join("whisper-burn-cli-t26");
    std::fs::create_dir_all(&dir).unwrap();
    let flac = whisper_burn::download::download_to(
        "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        &dir.join("jfk.flac"),
        false,
    )
    .unwrap();

    let out = Command::new(EXE)
        .args(["--progress", "--language", "en", "--device", "cpu"])
        .arg(&flac)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    for needle in [
        "weights:",           // model/weights stage
        "audio:",             // audio stage
        "decode: window 1/1", // window progress
    ] {
        assert!(
            stderr.contains(needle),
            "stderr missing {needle:?}:\n{stderr}"
        );
    }
    // streamed segment line with a timestamp bracket + arrow
    assert!(
        stderr.contains("[00:00.") && stderr.contains("-->"),
        "no streamed segment line on stderr:\n{stderr}"
    );
}
