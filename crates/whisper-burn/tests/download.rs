#![cfg(feature = "weights")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use whisper_burn::config::ModelSize;
use whisper_burn::download::{
    cache_dir, download_checkpoint, download_to, download_to_with_progress, repo_base,
};

/// Serializes tests that touch the process-global environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Spawns a tiny HTTP server on a free localhost port. Each connection is
/// answered with `status` (or dropped entirely when `status == 0`), then the
/// connection is closed. Returns the port and a connection counter.
fn spawn_server(status: u16, body: &'static [u8], times: usize) -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    thread::spawn(move || {
        for _ in 0..times {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            h.fetch_add(1, Ordering::SeqCst);
            if status == 0 {
                drop(stream);
                continue;
            }
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    (port, hits)
}

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("wburn_dl_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn repo_base_points_to_hf_resolve_main() {
    assert_eq!(
        repo_base(ModelSize::Tiny),
        "https://huggingface.co/openai/whisper-tiny/resolve/main"
    );
    assert_eq!(
        repo_base(ModelSize::LargeV3),
        "https://huggingface.co/openai/whisper-large-v3/resolve/main"
    );
}

#[test]
fn cache_dir_honors_env_override_and_default_layout() {
    let _guard = ENV_LOCK.lock().unwrap();
    let cache = tmp_dir("cache_env");
    let prev = std::env::var_os("WHISPER_BURN_CACHE");
    unsafe {
        std::env::set_var("WHISPER_BURN_CACHE", &cache);
        let overridden = cache_dir(ModelSize::Tiny).unwrap();
        assert!(overridden.ends_with(Path::new("openai").join("whisper-tiny")));
        std::env::remove_var("WHISPER_BURN_CACHE");
    }
    let default = cache_dir(ModelSize::Tiny).unwrap();
    assert!(
        default.ends_with(
            Path::new(".cache")
                .join("whisper-burn")
                .join("openai")
                .join("whisper-tiny")
        ),
        "unexpected default cache path: {}",
        default.display()
    );
    match prev {
        Some(v) => unsafe { std::env::set_var("WHISPER_BURN_CACHE", v) },
        None => unsafe { std::env::remove_var("WHISPER_BURN_CACHE") },
    }
}

#[test]
fn download_to_writes_file_atomically() {
    const BODY: &[u8] = b"ATOMIC-WEIGHTS-BYTES";
    let (port, _hits) = spawn_server(200, BODY, 1);
    let dir = tmp_dir("atomic");
    let dest = dir.join("model.safetensors");
    let url = format!("http://127.0.0.1:{port}/model.safetensors");

    let got = download_to(&url, &dest, false).unwrap();

    assert_eq!(got, dest);
    assert_eq!(std::fs::read(&dest).unwrap(), BODY);
    assert!(
        !dest.with_extension("safetensors.part").exists(),
        "no .part file should remain after success"
    );
}

#[test]
fn download_to_returns_http_status_as_error() {
    let (port, hits) = spawn_server(404, b"not found", 1);
    let dir = tmp_dir("status404");
    let dest = dir.join("config.json");
    let url = format!("http://127.0.0.1:{port}/config.json");

    let err = download_to(&url, &dest, false).unwrap_err();

    assert!(
        err.to_string().contains("404"),
        "error should surface the HTTP status: {err}"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    assert!(!dest.exists());
}

#[test]
fn download_to_retries_transport_errors() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    // First connection: drop without any response (transport error).
    // Second connection: serve the body.
    const BODY: &[u8] = b"RETRIED-BODY";
    thread::spawn(move || {
        for i in 0..2usize {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            h.fetch_add(1, Ordering::SeqCst);
            if i == 0 {
                drop(stream);
                continue;
            }
            let head = format!(
                "HTTP/1.1 200 X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                BODY.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(BODY);
        }
    });

    let dir = tmp_dir("retry");
    let dest = dir.join("model.safetensors");
    let url = format!("http://127.0.0.1:{port}/model.safetensors");

    let got = download_to(&url, &dest, false).unwrap();

    assert_eq!(got, dest);
    assert_eq!(std::fs::read(&dest).unwrap(), BODY);
    assert_eq!(
        hits.load(Ordering::SeqCst),
        2,
        "should retry once after transport failure"
    );
}

#[test]
fn download_to_gives_up_after_three_transport_failures() {
    let (port, hits) = spawn_server(0, b"", 3);
    let dir = tmp_dir("give_up");
    let dest = dir.join("model.safetensors");
    let url = format!("http://127.0.0.1:{port}/model.safetensors");

    let err = download_to(&url, &dest, false).unwrap_err();

    assert!(
        err.to_string().contains("download"),
        "unexpected error: {err}"
    );
    assert!(hits.load(Ordering::SeqCst) <= 3);
    assert!(!dest.exists());
}

#[test]
fn download_to_skips_when_not_overwrite_and_exists() {
    const PREV: &[u8] = b"EXISTING-BYTES";
    let dir = tmp_dir("skip_existing");
    let dest = dir.join("config.json");
    std::fs::write(&dest, PREV).unwrap();

    // Port with no listener: reaching it would be a transport error.
    let err_url = "http://127.0.0.1:1/config.json";
    let got = download_to(err_url, &dest, false).unwrap();

    assert_eq!(got, dest);
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        PREV,
        "existing file untouched"
    );
}

/// A lazily-accepted-size build; kept small so the callback sees every byte.
#[test]
fn download_to_with_progress_streams_bytes_and_total() {
    const BODY: &[u8] = b"PROGRESS-WEIGHTS-BODY-0123456789";
    let (port, _hits) = spawn_server(200, BODY, 1);
    let dir = tmp_dir("progress");
    let dest = dir.join("model.safetensors");
    let url = format!("http://127.0.0.1:{port}/model.safetensors");

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = seen.clone();
    let cb: whisper_burn::download::DownloadCallback = std::sync::Arc::new(
        move |downloaded: u64, total: Option<u64>| sink.lock().unwrap().push((downloaded, total)),
    );

    download_to_with_progress(&url, &dest, false, Some(cb)).unwrap();

    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty(), "progress callback never invoked");
    assert_eq!(seen.last(), Some(&(BODY.len() as u64, Some(BODY.len() as u64))));
    assert!(
        seen.windows(2).all(|w| w[0].0 <= w[1].0),
        "bytes must be monotonic: {seen:?}"
    );
}

#[test]
#[ignore = "network"]
fn downloads_tiny_checkpoint() {
    let dir = tmp_dir("tiny_network");
    let path = download_checkpoint(ModelSize::Tiny, &dir, true, true).unwrap();
    assert!(path.ends_with("model.safetensors"));
    assert!(dir.join("model.safetensors").exists());
    assert!(dir.join("config.json").exists());
}
