use dictation_service::chord::{Chord, ChordTracker, HotkeyEvent::*};

fn t() -> ChordTracker {
    ChordTracker::new(Chord::parse("ctrl+space").unwrap())
}

#[test]
fn press_then_release() {
    let mut c = t();
    assert_eq!(c.feed("ctrl", true), None);
    assert_eq!(c.feed("space", true), Some(Pressed));
    assert_eq!(c.feed("space", false), Some(Released));
}

#[test]
fn release_by_either_key() {
    let mut c = t();
    c.feed("ctrl", true);
    c.feed("space", true);
    assert_eq!(c.feed("ctrl", false), Some(Released));
    assert_eq!(c.feed("space", false), None);
}

#[test]
fn autorepeat_does_not_retrigger() {
    let mut c = t();
    c.feed("ctrl", true);
    assert_eq!(c.feed("space", true), Some(Pressed));
    assert_eq!(c.feed("space", true), None);
}

#[test]
fn space_before_ctrl_also_works() {
    let mut c = t();
    assert_eq!(c.feed("space", true), None);
    assert_eq!(c.feed("ctrl", true), Some(Pressed));
}

#[test]
fn unrelated_keys_are_ignored() {
    let mut c = t();
    assert_eq!(c.feed("a", true), None);
}

#[test]
fn parse_rejects_empty_and_normalizes_case() {
    assert!(Chord::parse("").is_err());
    assert!(Chord::parse("Ctrl + SPACE").is_ok());
    assert_eq!(
        Chord::parse("Ctrl + SPACE").unwrap(),
        Chord::parse("space+ctrl").unwrap()
    );
}

#[test]
fn can_press_again_after_release() {
    let mut c = t();
    c.feed("ctrl", true);
    c.feed("space", true);
    c.feed("space", false);
    assert_eq!(c.feed("space", true), Some(Pressed));
}

#[test]
fn single_key_chord_works() {
    let mut c = ChordTracker::new(Chord::parse("f9").unwrap());
    assert_eq!(c.feed("f9", true), Some(Pressed));
    assert_eq!(c.feed("f9", false), Some(Released));
}
