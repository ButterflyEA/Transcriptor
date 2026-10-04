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
