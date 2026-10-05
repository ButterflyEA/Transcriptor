use dictation_service::audio::Recording;
use dictation_service::controller::{Action, Controller, Input, Status};

fn rec(ms: u64) -> Recording {
    Recording {
        samples: vec![0.1; (16 * ms) as usize],
        sample_rate: 16_000,
    }
}

fn ready() -> Controller {
    let mut c = Controller::new(3);
    c.handle(Input::EngineReady);
    c
}

fn has(a: &[Action], f: impl Fn(&Action) -> bool) -> bool {
    a.iter().any(f)
}

fn record(c: &mut Controller, ms: u64) -> Vec<Action> {
    c.handle(Input::Press);
    c.handle(Input::Release);
    c.handle(Input::Captured(rec(ms)))
}

#[test]
fn press_is_ignored_while_loading() {
    let mut c = Controller::new(3);
    let a = c.handle(Input::Press);
    assert!(!has(&a, |x| matches!(x, Action::StartCapture)));
    assert!(has(&a, |x| matches!(x, Action::Notify(_))));
}

#[test]
fn press_release_transcribe_paste_flow() {
    let mut c = ready();
    assert!(has(&c.handle(Input::Press), |x| matches!(
        x,
        Action::StartCapture
    )));
    assert!(has(&c.handle(Input::Release), |x| matches!(
        x,
        Action::StopCapture
    )));
    let a = c.handle(Input::Captured(rec(1000)));
    assert!(has(&a, |x| matches!(x, Action::Transcribe(_))));
    let a = c.handle(Input::Transcribed(Ok("hello".into())));
    assert!(has(&a, |x| matches!(x, Action::Paste(t) if t == "hello")));
    assert!(has(&a, |x| matches!(x, Action::SetStatus(Status::Idle))));
}

#[test]
fn short_recordings_are_discarded() {
    let mut c = ready();
    let a = record(&mut c, 100);
    assert!(!has(&a, |x| matches!(x, Action::Transcribe(_))));
}

#[test]
fn press_while_transcribing_queues_in_order() {
    let mut c = ready();
    record(&mut c, 1000);
    assert!(has(&c.handle(Input::Press), |x| matches!(
        x,
        Action::StartCapture
    )));
    c.handle(Input::Release);
    let a = c.handle(Input::Captured(rec(2000)));
    assert!(!has(&a, |x| matches!(x, Action::Transcribe(_))));
    let a = c.handle(Input::Transcribed(Ok("one".into())));
    assert!(has(&a, |x| matches!(x, Action::Paste(t) if t == "one")));
    assert!(has(
        &a,
        |x| matches!(x, Action::Transcribe(r) if r.duration_ms() == 2000)
    ));
}

#[test]
fn queue_is_capped() {
    let mut c = Controller::new(1);
    c.handle(Input::EngineReady);
    record(&mut c, 1000); // running
    record(&mut c, 1000); // queued 1/1
    let a = record(&mut c, 1000); // dropped
    assert!(has(
        &a,
        |x| matches!(x, Action::Notify(m) if m.contains("queue"))
    ));
}

#[test]
fn empty_transcript_is_not_pasted() {
    let mut c = ready();
    record(&mut c, 1000);
    let a = c.handle(Input::Transcribed(Ok("   ".into())));
    assert!(!has(&a, |x| matches!(x, Action::Paste(_))));
}

#[test]
fn transcription_failure_notifies_and_moves_on() {
    let mut c = ready();
    record(&mut c, 1000);
    let a = c.handle(Input::Transcribed(Err("boom".into())));
    assert!(has(&a, |x| matches!(x, Action::Notify(_))));
}

#[test]
fn double_press_and_stray_release_are_ignored() {
    let mut c = ready();
    c.handle(Input::Press);
    assert!(c.handle(Input::Press).is_empty());
    c.handle(Input::Release);
    c.handle(Input::Captured(rec(1000)));
    assert!(c.handle(Input::Release).is_empty());
}

#[test]
fn status_is_recording_while_busy() {
    let mut c = ready();
    record(&mut c, 1000);
    let a = c.handle(Input::Press);
    assert!(has(&a, |x| matches!(
        x,
        Action::SetStatus(Status::Recording)
    )));
}

#[test]
fn is_recording_tracks_press_and_capture() {
    let mut c = ready();
    assert!(!c.is_recording());
    c.handle(Input::Press);
    assert!(c.is_recording());
    c.handle(Input::Release);
    c.handle(Input::Captured(rec(1000)));
    assert!(!c.is_recording());
}

#[test]
fn set_loading_blocks_presses_until_ready_again() {
    let mut c = ready();
    let a = c.set_loading();
    assert!(has(&a, |x| matches!(x, Action::SetStatus(Status::Loading))));
    assert!(has(&c.handle(Input::Press), |x| matches!(
        x,
        Action::Notify(_)
    )));
    c.handle(Input::EngineReady);
    assert!(has(&c.handle(Input::Press), |x| matches!(
        x,
        Action::StartCapture
    )));
}

#[test]
fn status_names() {
    assert_eq!(Status::Idle.as_str(), "idle");
    assert_eq!(Status::Recording.as_str(), "recording");
}

#[test]
fn engine_failure_keeps_error_status_and_blocks_recording() {
    let mut c = Controller::new(3);
    let a = c.handle(Input::EngineFailed("boom".into()));
    assert!(has(&a, |x| matches!(x, Action::SetStatus(Status::Error))));
    let a = c.handle(Input::Press);
    assert!(!has(&a, |x| matches!(x, Action::StartCapture)));
    assert!(!c.is_recording());
}

#[test]
fn reload_after_engine_failure_recovers() {
    let mut c = Controller::new(3);
    c.handle(Input::EngineFailed("boom".into()));
    c.set_loading();
    c.handle(Input::EngineReady);
    let a = c.handle(Input::Press);
    assert!(has(&a, |x| matches!(x, Action::StartCapture)));
}
