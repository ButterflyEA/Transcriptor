use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use dictation_service::audio::LiveTap;
use dictation_service::stream::*;
use dictation_service::transcript::Edit;

const RATE: u32 = 16_000;

fn tone(secs: f32) -> Vec<f32> {
    (0..(RATE as f32 * secs) as usize)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE as f32).sin())
        .collect()
}

fn params(window_secs: f32, interval_ms: u64) -> StreamParams {
    StreamParams {
        window_samples: (RATE as f32 * window_secs) as usize,
        overlap_samples: 0,
        interval: Duration::from_millis(interval_ms),
        lock_words: 2,
        rms_threshold: 0.01,
    }
}

fn tap(samples: Vec<f32>) -> LiveTap {
    LiveTap::new(Arc::new(Mutex::new(samples)), RATE)
}

/// Runs the loop on a thread for `run_ms`, returning the events it produced.
fn run_for<D: WindowDecoder + Send + 'static>(
    decoder: D,
    p: StreamParams,
    tap: LiveTap,
    run_ms: u64,
) -> Vec<StreamEvent> {
    let stop = Arc::new(AtomicBool::new(false));
    let events = Arc::new(Mutex::new(Vec::new()));
    let (s, e) = (stop.clone(), events.clone());
    let h = thread::spawn(move || {
        StreamLoop::new(decoder, p).run(tap, 7, s, |ev| e.lock().unwrap().push(ev));
    });
    thread::sleep(Duration::from_millis(run_ms));
    stop.store(true, Ordering::SeqCst);
    h.join().unwrap();
    events.lock().unwrap().clone()
}

fn typed(events: &[StreamEvent]) -> String {
    let mut buf = String::new();
    for ev in events {
        if let StreamEvent::Edit { edit, .. } = ev {
            match edit {
                Edit::Append(t) => buf.push_str(t),
                Edit::Replace { backspaces, text } => {
                    for _ in 0..*backspaces {
                        buf.pop();
                    }
                    buf.push_str(text);
                }
            }
        }
    }
    buf
}

struct Script {
    calls: Arc<Mutex<Vec<usize>>>,
    texts: Vec<&'static str>,
    delay: Duration,
}

impl WindowDecoder for Script {
    fn decode(&self, pcm: &[f32], _: &Arc<AtomicBool>) -> Result<String, String> {
        thread::sleep(self.delay);
        let mut calls = self.calls.lock().unwrap();
        calls.push(pcm.len());
        let i = (calls.len() - 1).min(self.texts.len() - 1);
        Ok(self.texts[i].to_string())
    }
}

fn script(texts: &[&'static str], delay_ms: u64) -> (Script, Arc<Mutex<Vec<usize>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    (
        Script {
            calls: calls.clone(),
            texts: texts.to_vec(),
            delay: Duration::from_millis(delay_ms),
        },
        calls,
    )
}

#[test]
fn growing_text_is_emitted_in_order_with_the_capture_id() {
    let (d, _) = script(
        &["שלום", "שלום עולם", "שלום עולם טוב", "שלום עולם טוב מאוד"],
        0,
    );
    let buf = Arc::new(Mutex::new(tone(1.0)));
    let t = LiveTap::new(buf.clone(), RATE);
    let grow = thread::spawn(move || {
        for _ in 0..4 {
            thread::sleep(Duration::from_millis(100));
            buf.lock().unwrap().extend(tone(0.2));
        }
    });
    let events = run_for(d, params(30.0, 20), t, 700);
    grow.join().unwrap();
    assert!(events.iter().all(|e| match e {
        StreamEvent::Edit { capture_id, .. } => *capture_id == 7,
        _ => true,
    }));
    assert_eq!(typed(&events), "שלום עולם טוב מאוד");
}

#[test]
fn passes_never_overlap_when_the_decoder_is_slow() {
    let (d, calls) = script(&["אחד"], 100);
    run_for(d, params(30.0, 10), tap(tone(2.0)), 450);
    // One audio length means one pass only: later passes see no new audio.
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[test]
fn silence_emits_nothing_and_is_never_decoded() {
    let (d, calls) = script(&["הזיה"], 0);
    let events = run_for(d, params(30.0, 20), tap(vec![0.0; 32_000]), 200);
    assert!(events.is_empty());
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn completed_windows_advance_and_are_never_longer_than_the_window() {
    let (d, calls) = script(&["אחד", "אחד שניים", "שלושה", "שלושה ארבעה"], 0);
    let events = run_for(d, params(1.0, 20), tap(tone(2.5)), 500);
    let calls = calls.lock().unwrap().clone();
    assert!(calls.iter().all(|&n| n <= 16_000), "{calls:?}");
    assert!(calls.contains(&16_000));
    assert!(typed(&events).starts_with("אחד"), "{}", typed(&events));
}

struct Blocking(Arc<Mutex<bool>>);
impl WindowDecoder for Blocking {
    fn decode(&self, _: &[f32], cancelled: &Arc<AtomicBool>) -> Result<String, String> {
        while !cancelled.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(5));
        }
        *self.0.lock().unwrap() = true;
        Err("cancelled".into())
    }
}

#[test]
fn stop_cancels_the_pass_in_flight_and_the_loop_returns() {
    let saw_cancel = Arc::new(Mutex::new(false));
    let events = run_for(
        Blocking(saw_cancel.clone()),
        params(30.0, 10),
        tap(tone(2.0)),
        200,
    );
    assert!(*saw_cancel.lock().unwrap());
    assert!(events.is_empty());
}

struct Slowish;
impl WindowDecoder for Slowish {
    fn decode(&self, _: &[f32], _: &Arc<AtomicBool>) -> Result<String, String> {
        thread::sleep(Duration::from_millis(400));
        Ok("מילה".into())
    }
}

#[test]
fn a_pass_slower_than_three_windows_reports_slow_once() {
    // Window of 1 s is the minimum; fake a tiny window of 0.5 s so 250 ms is not slow,
    // then use a 0.05 s window so 250 ms exceeds 3 windows.
    let p = StreamParams {
        window_samples: 1600,
        overlap_samples: 0,
        interval: Duration::from_millis(10),
        lock_words: 1,
        rms_threshold: 0.01,
    };
    let growing = Arc::new(Mutex::new(tone(1.0)));
    let t = LiveTap::new(growing.clone(), RATE);
    let stop = Arc::new(AtomicBool::new(false));
    let events = Arc::new(Mutex::new(Vec::new()));
    let (s, e) = (stop.clone(), events.clone());
    let h = thread::spawn(move || {
        StreamLoop::new(Slowish, p).run(t, 1, s, |ev| e.lock().unwrap().push(ev));
    });
    for _ in 0..4 {
        thread::sleep(Duration::from_millis(500));
        growing.lock().unwrap().extend(tone(0.2));
    }
    stop.store(true, Ordering::SeqCst);
    h.join().unwrap();
    let slow = events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| matches!(e, StreamEvent::Slow { .. }))
        .count();
    assert_eq!(slow, 1);
}
