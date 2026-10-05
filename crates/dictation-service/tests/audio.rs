use dictation_service::audio::{LiveTap, Recording, downmix_to_mono, push_capped};
use std::sync::{Arc, Mutex};

#[test]
fn mono_passthrough() {
    assert_eq!(downmix_to_mono(&[0.1, 0.2], 1), vec![0.1, 0.2]);
}

#[test]
fn stereo_is_averaged() {
    assert_eq!(downmix_to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
}

#[test]
fn trailing_partial_frame_is_dropped() {
    assert_eq!(downmix_to_mono(&[1.0, 1.0, 9.0], 2), vec![1.0]);
}

#[test]
fn duration_ms_from_rate() {
    let r = Recording {
        samples: vec![0.0; 8000],
        sample_rate: 16_000,
    };
    assert_eq!(r.duration_ms(), 500);
}

#[test]
fn zero_rate_has_zero_duration() {
    let r = Recording {
        samples: vec![0.0; 10],
        sample_rate: 0,
    };
    assert_eq!(r.duration_ms(), 0);
}

#[test]
fn peak_is_max_abs() {
    let r = Recording {
        samples: vec![0.1, -0.7, 0.3],
        sample_rate: 16_000,
    };
    assert_eq!(r.peak(), 0.7);
}

#[test]
fn push_capped_stops_at_max() {
    let mut v = Vec::new();
    push_capped(&mut v, &[1.0; 5], 3);
    assert_eq!(v.len(), 3);
    push_capped(&mut v, &[1.0; 5], 3);
    assert_eq!(v.len(), 3);
}

fn tap_with(n: usize) -> (LiveTap, Arc<Mutex<Vec<f32>>>) {
    let buf = Arc::new(Mutex::new((0..n).map(|i| i as f32).collect::<Vec<_>>()));
    (LiveTap::new(Arc::clone(&buf), 16_000), buf)
}

#[test]
fn tap_snapshots_whole_capture_then_only_new_samples() {
    let (tap, buf) = tap_with(1000);
    let (s, rate, total) = tap.snapshot_from(0);
    assert_eq!((s.len(), rate, total), (1000, 16_000, 1000));
    buf.lock().unwrap().extend(std::iter::repeat_n(0.5, 500));
    let (s, _, total) = tap.snapshot_from(1000);
    assert_eq!((s.len(), total), (500, 1500));
    assert_eq!(tap.len(), 1500);
}

#[test]
fn tap_offset_past_the_end_is_empty() {
    let (tap, _) = tap_with(10);
    let (s, _, total) = tap.snapshot_from(50);
    assert!(s.is_empty());
    assert_eq!(total, 10);
}

#[test]
fn cloned_tap_sees_later_pushes() {
    let (tap, buf) = tap_with(0);
    let other = tap.clone();
    assert!(other.is_empty());
    buf.lock().unwrap().push(1.0);
    assert_eq!(other.len(), 1);
}
