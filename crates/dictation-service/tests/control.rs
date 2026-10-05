use std::io::ErrorKind;
use std::thread;
use std::time::Duration;

use dictation_service::control::{Command, send_to, serve_on};

#[test]
fn parses_known_commands_case_insensitively() {
    assert_eq!(Command::parse("START\n"), Some(Command::Start));
    assert_eq!(Command::parse(" toggle "), Some(Command::Toggle));
}

#[test]
fn rejects_unknown() {
    assert_eq!(Command::parse("rm -rf"), None);
}

#[test]
fn round_trips_names() {
    for c in [
        Command::Start,
        Command::Stop,
        Command::Toggle,
        Command::Status,
        Command::Reload,
    ] {
        assert_eq!(Command::parse(c.as_line()), Some(c));
    }
}

fn test_name(tag: &str) -> String {
    let id = format!("dictation-test-{tag}-{}", std::process::id());
    if cfg!(windows) {
        id
    } else {
        std::env::temp_dir().join(id).to_string_lossy().into_owned()
    }
}

fn wait_for(name: &str) {
    for _ in 0..50 {
        if send_to(name, Command::Status).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("server did not come up");
}

#[test]
fn server_answers_commands_over_the_socket() {
    let name = test_name("roundtrip");
    let n = name.clone();
    thread::spawn(move || {
        let _ = serve_on(&n, |c| format!("got {}", c.as_line()));
    });
    wait_for(&name);
    assert_eq!(send_to(&name, Command::Toggle).unwrap(), "got toggle");
    assert_eq!(send_to(&name, Command::Reload).unwrap(), "got reload");
}

#[test]
fn second_server_is_refused() {
    let name = test_name("single");
    let n = name.clone();
    thread::spawn(move || {
        let _ = serve_on(&n, |_| "ok".into());
    });
    wait_for(&name);
    let err = serve_on(&name, |_| "ok".into()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::AddrInUse, "{err}");
}

#[test]
fn client_errors_when_no_server_is_running() {
    let name = test_name("none");
    assert!(send_to(&name, Command::Status).is_err());
}
