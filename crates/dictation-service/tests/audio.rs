use dictation_service::audio::{Recording, downmix_to_mono, push_capped};

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
