use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dictation_service::audio::{LiveTap, Recorder, Recording};
use dictation_service::config::Config;
use dictation_service::controller::{Input, Status};
use dictation_service::output::{PasteOutcome, Paster};
use dictation_service::runtime::{RuntimeMsg, spawn_with, spawn_with_typer};
use dictation_service::typer::Typer;
use whisper_burn::audio::decode::decode_to_mono_f32;

struct FakeRecorder(Recording);
impl Recorder for FakeRecorder {
    fn start(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn stop(&mut self) -> Result<Recording, String> {
        Ok(self.0.clone())
    }
}

struct FakePaster(Arc<Mutex<Vec<String>>>);
impl Paster for FakePaster {
    fn paste(&mut self, text: &str) -> Result<PasteOutcome, String> {
        self.0.lock().unwrap().push(text.to_owned());
        Ok(PasteOutcome::Pasted)
    }
}

#[test]
fn missing_model_reports_failure_and_never_records() {
    let dir = std::env::temp_dir().join("dictation-no-model");
    let cfg = Config {
        model_dir: Some(dir),
        ..Config::default()
    };
    let (ntx, nrx) = mpsc::channel();
    let pasted = Arc::new(Mutex::new(Vec::new()));
    let rt = spawn_with(
        cfg,
        Box::new(FakeRecorder(Recording {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
        })),
        Box::new(FakePaster(pasted.clone())),
        Box::new(|_| {}),
        Box::new(move |m| {
            let _ = ntx.send(m);
        }),
    );
    let msg = nrx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(msg.contains("model not found"), "{msg}");
    rt.shutdown();
    assert!(pasted.lock().unwrap().is_empty());
}

/// Needs DICTATION_MODEL_DIR and the Hebrew sample clip at the repo root.
#[test]
#[ignore]
fn end_to_end_hebrew_dictation_pastes_text() {
    let model_dir = PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_16000_mono.wav");
    let (samples, sample_rate) = decode_to_mono_f32(&wav).unwrap();

    let cfg = Config {
        model_dir: Some(model_dir),
        language: "he".into(),
        ..Config::default()
    };
    let (stx, srx) = mpsc::channel();
    let pasted = Arc::new(Mutex::new(Vec::new()));
    let rt = spawn_with(
        cfg,
        Box::new(FakeRecorder(Recording {
            samples,
            sample_rate,
        })),
        Box::new(FakePaster(pasted.clone())),
        Box::new(move |s| {
            let _ = stx.send(s);
        }),
        Box::new(|m| eprintln!("notify: {m}")),
    );
    let wait_for = |want: Status| loop {
        let s = srx.recv_timeout(Duration::from_secs(120)).expect("status");
        if s == want {
            break;
        }
    };
    wait_for(Status::Idle);
    rt.send(Input::Press);
    wait_for(Status::Recording);
    rt.send(Input::Release);
    wait_for(Status::Transcribing);
    wait_for(Status::Idle);
    rt.shutdown();

    let got = pasted.lock().unwrap().clone();
    assert_eq!(got.len(), 1, "{got:?}");
    assert!(
        got[0].contains("\u{5e9}\u{5dc}\u{5d5}\u{5dd}"),
        "{}",
        got[0]
    );
}

#[test]
#[ignore]
fn toggle_and_reload_work_with_the_real_model() {
    let model_dir = PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_16000_mono.wav");
    let (samples, sample_rate) = decode_to_mono_f32(&wav).unwrap();
    let cfg = Config {
        model_dir: Some(model_dir),
        language: "he".into(),
        ..Config::default()
    };
    let (stx, srx) = mpsc::channel();
    let pasted = Arc::new(Mutex::new(Vec::new()));
    let rt = spawn_with(
        cfg.clone(),
        Box::new(FakeRecorder(Recording {
            samples,
            sample_rate,
        })),
        Box::new(FakePaster(pasted.clone())),
        Box::new(move |s| {
            let _ = stx.send(s);
        }),
        Box::new(|m| eprintln!("notify: {m}")),
    );
    let wait_for = |want: Status| loop {
        let s = srx.recv_timeout(Duration::from_secs(120)).expect("status");
        if s == want {
            break;
        }
    };
    wait_for(Status::Idle);

    // Reload with an unchanged model: goes Loading then straight back to Idle.
    rt.tx.send(RuntimeMsg::Reload(Box::new(cfg))).unwrap();
    wait_for(Status::Loading);
    wait_for(Status::Idle);

    rt.tx.send(RuntimeMsg::Toggle).unwrap();
    wait_for(Status::Recording);
    assert_eq!(rt.status(), Status::Recording);
    rt.tx.send(RuntimeMsg::Toggle).unwrap();
    wait_for(Status::Transcribing);
    wait_for(Status::Idle);
    rt.shutdown();
    assert_eq!(pasted.lock().unwrap().len(), 1);
}

struct TapRecorder(Recording);
impl Recorder for TapRecorder {
    fn start(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn stop(&mut self) -> Result<Recording, String> {
        Ok(self.0.clone())
    }
    fn tap(&self) -> Option<LiveTap> {
        Some(LiveTap::new(
            Arc::new(Mutex::new(self.0.samples.clone())),
            self.0.sample_rate,
        ))
    }
}

/// In stream mode a second press ends the recording and the release is ignored.
#[test]
#[ignore]
fn stream_mode_toggles_on_press_and_still_pastes_the_batch_text() {
    let model_dir = PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_16000_mono.wav");
    let (samples, sample_rate) = decode_to_mono_f32(&wav).unwrap();
    let cfg = Config {
        model_dir: Some(model_dir),
        language: "he".into(),
        stream: true,
        ..Config::default()
    };
    let (stx, srx) = mpsc::channel();
    let pasted = Arc::new(Mutex::new(Vec::new()));
    let rt = spawn_with(
        cfg,
        Box::new(TapRecorder(Recording {
            samples,
            sample_rate,
        })),
        Box::new(FakePaster(pasted.clone())),
        Box::new(move |s| {
            let _ = stx.send(s);
        }),
        Box::new(|m| eprintln!("notify: {m}")),
    );
    let wait_for = |want: Status| loop {
        let s = srx.recv_timeout(Duration::from_secs(120)).expect("status");
        if s == want {
            break;
        }
    };
    wait_for(Status::Idle);
    rt.send(Input::Press);
    wait_for(Status::Recording);
    rt.send(Input::Release);
    std::thread::sleep(Duration::from_secs(6));
    assert_eq!(rt.status(), Status::Recording);
    rt.send(Input::Press);
    wait_for(Status::Transcribing);
    wait_for(Status::Idle);
    rt.shutdown();
    assert_eq!(pasted.lock().unwrap().len(), 1);
}
struct LogPaster(Arc<Mutex<Vec<String>>>);
impl Paster for LogPaster {
    fn paste(&mut self, text: &str) -> Result<PasteOutcome, String> {
        self.0.lock().unwrap().push(format!("paste:{text}"));
        Ok(PasteOutcome::Pasted)
    }
    fn set_clipboard(&mut self, text: &str) -> Result<(), String> {
        self.0.lock().unwrap().push(format!("clip:{text}"));
        Ok(())
    }
}

/// Mimics a text box: applies typed text and backspaces to a buffer.
struct BufTyper(Arc<Mutex<String>>);
impl Typer for BufTyper {
    fn type_text(&mut self, s: &str) -> Result<(), String> {
        self.0.lock().unwrap().push_str(s);
        Ok(())
    }
    fn backspace(&mut self, n: usize) -> Result<(), String> {
        let mut b = self.0.lock().unwrap();
        for _ in 0..n {
            b.pop();
        }
        Ok(())
    }
    fn supports_backspace(&self) -> bool {
        true
    }
}

/// The text box ends up holding exactly the final text, which is also on the clipboard.
#[test]
#[ignore]
fn live_typing_ends_with_the_final_text_in_the_box_and_clipboard() {
    let model_dir = PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_16000_mono.wav");
    let (samples, sample_rate) = decode_to_mono_f32(&wav).unwrap();
    let cfg = Config {
        model_dir: Some(model_dir),
        language: "he".into(),
        stream: true,
        ..Config::default()
    };
    let (stx, srx) = mpsc::channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let boxed = Arc::new(Mutex::new(String::new()));
    let rt = spawn_with_typer(
        cfg,
        Box::new(TapRecorder(Recording {
            samples,
            sample_rate,
        })),
        Box::new(LogPaster(log.clone())),
        Box::new(BufTyper(boxed.clone())),
        Box::new(move |s| {
            let _ = stx.send(s);
        }),
        Box::new(|m| eprintln!("notify: {m}")),
    );
    let wait_for = |want: Status| loop {
        let s = srx.recv_timeout(Duration::from_secs(120)).expect("status");
        if s == want {
            break;
        }
    };
    wait_for(Status::Idle);
    rt.send(Input::Press);
    wait_for(Status::Recording);
    std::thread::sleep(Duration::from_secs(6));
    let mid = boxed.lock().unwrap().clone();
    assert!(!mid.is_empty(), "nothing was typed while recording");
    rt.send(Input::Press);
    wait_for(Status::Transcribing);
    wait_for(Status::Idle);
    rt.shutdown();
    let log = log.lock().unwrap().clone();
    assert_eq!(log.len(), 1, "{log:?}");
    let final_text = log[0].strip_prefix("clip:").expect("clipboard only");
    assert_eq!(*boxed.lock().unwrap(), final_text);
}
