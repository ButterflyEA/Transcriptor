use dictation_service::checksum::{FileCheck, parse_sums, sha256_file, verify_dir};

fn temp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

#[test]
fn hashes_a_file() {
    let d = temp("dictation-sum-a");
    std::fs::write(d.join("f"), b"abc").unwrap();
    assert_eq!(sha256_file(&d.join("f")).unwrap(), ABC);
}

#[test]
fn parses_sha256sum_lines() {
    let text = format!("{ABC}  model.safetensors\n{ABC} *config.json\nnonsense\n");
    let got = parse_sums(&text);
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].1, "config.json");
}

#[test]
fn verify_dir_reports_ok_mismatch_and_missing() {
    let d = temp("dictation-sum-b");
    std::fs::write(d.join("good"), b"abc").unwrap();
    std::fs::write(d.join("bad"), b"abd").unwrap();
    std::fs::write(
        d.join("SHA256SUMS"),
        format!("{ABC}  good\n{ABC}  bad\n{ABC}  gone\n"),
    )
    .unwrap();
    let r = verify_dir(&d).unwrap();
    assert_eq!(r[0].1, FileCheck::Ok);
    assert!(matches!(r[1].1, FileCheck::Mismatch { .. }));
    assert_eq!(r[2].1, FileCheck::Missing);
}

#[test]
fn verify_dir_without_sums_file_is_an_error() {
    assert!(verify_dir(&temp("dictation-sum-c")).is_err());
}
