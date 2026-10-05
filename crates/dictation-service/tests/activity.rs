use dictation_service::activity::*;

#[test]
fn unlocked_sync_corrects_the_tail() {
    assert_eq!(plan_sync("שלום עול", "שלום עולם", false), (0, "ם".into()));
    assert_eq!(plan_sync("abc.", "abc,d", false), (1, ",d".into()));
    assert_eq!(plan_sync("abc", "abc", false), (0, String::new()));
}

#[test]
fn locked_sync_only_appends() {
    assert_eq!(plan_sync("abc", "abcd", true), (0, "d".into()));
    assert_eq!(plan_sync("abc.", "abc,d", true), (0, String::new()));
}

#[test]
fn backspaces_count_characters_not_bytes() {
    assert_eq!(plan_sync("שלום", "של", false), (2, String::new()));
}

#[test]
fn our_own_typing_and_its_echo_are_not_user_activity() {
    let t0 = now_ms();
    begin_typing();
    note_user_input_at(now_ms() + 5);
    assert!(!user_active_since(t0));
    end_typing();
    note_user_input_at(now_ms() + 50);
    assert!(!user_active_since(t0));
    note_user_input_at(now_ms() + 1000);
    assert!(user_active_since(t0));
}
