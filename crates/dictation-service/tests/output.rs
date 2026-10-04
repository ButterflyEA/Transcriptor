use dictation_service::output::pick_paste_tool;

#[test]
fn prefers_wtype() {
    assert_eq!(pick_paste_tool(|_| true), Some("wtype"));
}

#[test]
fn falls_back_to_ydotool() {
    assert_eq!(pick_paste_tool(|t| t == "ydotool"), Some("ydotool"));
}

#[test]
fn none_when_missing() {
    assert_eq!(pick_paste_tool(|_| false), None);
}
