use dictation_service::hotkey::rdev_backend::key_name;
use rdev::Key;

#[test]
fn modifiers_collapse_left_and_right() {
    assert_eq!(key_name(Key::ControlLeft).as_deref(), Some("ctrl"));
    assert_eq!(key_name(Key::ControlRight).as_deref(), Some("ctrl"));
    assert_eq!(key_name(Key::ShiftRight).as_deref(), Some("shift"));
    assert_eq!(key_name(Key::AltGr).as_deref(), Some("alt"));
    assert_eq!(key_name(Key::MetaLeft).as_deref(), Some("meta"));
}

#[test]
fn common_keys_use_bare_names() {
    assert_eq!(key_name(Key::Space).as_deref(), Some("space"));
    assert_eq!(key_name(Key::KeyA).as_deref(), Some("a"));
    assert_eq!(key_name(Key::Num7).as_deref(), Some("7"));
    assert_eq!(key_name(Key::F9).as_deref(), Some("f9"));
}

#[test]
fn unknown_keys_are_ignored() {
    assert_eq!(key_name(Key::Unknown(999)), None);
}
