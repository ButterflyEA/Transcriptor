use dictation_service::controller::{Input, Status};
use dictation_service::runtime::{RuntimeMsg, blocked_by_pause};
use dictation_service::tray::{icon_rgba, tooltip};

#[test]
fn every_status_has_a_distinct_valid_icon() {
    let all = [
        Status::Loading,
        Status::Idle,
        Status::Recording,
        Status::Transcribing,
        Status::Error,
    ];
    let icons: Vec<_> = all.iter().map(|s| icon_rgba(*s)).collect();
    for i in &icons {
        assert_eq!(i.len(), 32 * 32 * 4);
        assert!(i.chunks(4).any(|p| p[3] == 255));
    }
    for a in 0..icons.len() {
        for b in a + 1..icons.len() {
            assert_ne!(icons[a], icons[b]);
        }
    }
}

#[test]
fn tooltip_names_the_status() {
    assert_eq!(tooltip(Status::Recording), "Dictation: recording");
}

#[test]
fn pause_blocks_new_recordings_only() {
    assert!(blocked_by_pause(true, &RuntimeMsg::Input(Input::Press)));
    assert!(blocked_by_pause(true, &RuntimeMsg::Toggle));
    assert!(!blocked_by_pause(true, &RuntimeMsg::Input(Input::Release)));
    assert!(!blocked_by_pause(false, &RuntimeMsg::Input(Input::Press)));
}
