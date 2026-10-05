use dictation_service::stitch::stitch;

fn v(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn drops_duplicated_overlap() {
    assert_eq!(
        stitch(&v("אני גר בגבעת"), &v("בגבעת אבני ואני"), 4).join(" "),
        "אני גר בגבעת אבני ואני"
    );
}

#[test]
fn no_overlap_just_concatenates() {
    assert_eq!(stitch(&v("א"), &v("ב"), 4), v("א ב"));
}

#[test]
fn prefers_the_longest_overlap_and_ignores_punctuation() {
    assert_eq!(stitch(&v("א ב ג"), &v("ב, ג ד"), 4), v("א ב, ג ד"));
}

#[test]
fn never_drops_more_than_the_limit() {
    assert_eq!(stitch(&v("א ב ג"), &v("א ב ג ד"), 2), v("א ב ג א ב ג ד"));
}

#[test]
fn empty_sides_are_fine() {
    assert_eq!(stitch(&[], &v("א"), 3), v("א"));
    assert_eq!(stitch(&v("א"), &[], 3), v("א"));
}