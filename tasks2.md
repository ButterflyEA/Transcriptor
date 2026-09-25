# Transcriptor Implementation Tasks (`tasks2.md`)

**For agentic workers:** implement this plan task-by-task with TDD (RED →
GREEN → commit per task). Each task is independently testable. Run the
regression gate (`cargo test --workspace` + `cargo clippy --workspace
--all-targets`) after every task and keep it green; commit at each task's
checkbox step.

Design spec: `docs/superpowers/specs/2026-09-22-transcriptor-design.md`

## Global Constraints

- Workspace: edition `2024`, resolver `2`. Add `"crates/transcriptor/src-tauri"`
  as a workspace member only from **T7** (it pulls in the Tauri build).
- Dependency pins: `tauri = "2"`, `tauri-build = "2"`,
  `tauri-plugin-dialog = "2"`, `docx-rs = "0.3"`, `printpdf = "0.8"`, `zip = "2"`
  (dev-only), `burn 0.21`, `serde 1`, `serde_json 1`. Do not accept major
  version bumps without re-reading the crate's API.
- Whisper model defaults (match the CLI): model `tiny`, task
  `transcribe`, beam size `0` (greedy), device `wgpu` with CPU fallback.
- Printpdf is pinned to the 0.8 API: `doc.add_external_font(&bytes)` and
  `layer.use_text(text, font_size, Mm(x), Mm(y), &font)`. SafeTensors F16/BF16
  handling in the library must not change (existing safetensors tests pin it).
- Windows packaging targets: `nsis` and `msi` (the spec's "Windows only first").
- No Windows-only APIs in app code except the single font-path candidate list
  in `format::pdf::load_font` (falls back gracefully; documented in-code).
- Language list for dropdowns: `whisper_burn::decoding::lang::language_window(n_vocab)`
  (`n_vocab = 51866`, the large-v3 count) — that function already exists.
- Every user-facing error text must render in the UI without crashing (map
  `whisper_burn::Error` → `AppError` in one place).
- Commit style: `feat:` / `fix:` / `docs:` prefix with a short `(whisper-burn)` /
  `(transcriptor)` / `(whisper-burn-cli)` scope tail, matching the repo history.

---

## T1: Progress events + cancellation in `whisper-burn`

**User story:** As the app, I want typed stage/segment events streamed from the
pipeline and a clean stop, so the UI can render progress live and cancel a run.

**Files:**
- Modify: `crates/whisper-burn/src/transcribe.rs`
- Test: `crates/whisper-burn/tests/transcribe.rs` (network module)

**Interfaces:**
- Produces:
  - `pub enum StageKind { Download, Weights, Audio, Decode, Done }` (`Clone, Copy, Debug, PartialEq, Eq`)
  - `pub enum ProgressUpdate { Stage { kind: StageKind, message: String, fraction: Option<f32> }, Segment(TranscriptionSegment) }` (`Clone, Debug, PartialEq`)
  - `pub type ProgressCallback = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>`
  - `TranscriptionOptions.progress: Option<ProgressCallback>`
  - `TranscriptionOptions.cancelled: Arc<AtomicBool>`

- [ ] **Step 1: Write the failing test**

Append to the `network` module in `crates/whisper-burn/tests/transcribe.rs`:

```rust
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use whisper_burn::transcribe::{ProgressUpdate, StageKind};

#[test]
#[ignore = "network"]
fn transcribe_streams_decode_segments_and_done_in_order() {
    let dev = NdArrayDevice::default();
    let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    let pcm = jfk_pcm();
    let events = Arc::new(Mutex::new(Vec::<ProgressUpdate>::new()));
    let sink = events.clone();
    let cb: Arc<dyn Fn(&ProgressUpdate) + Send + Sync> =
        Arc::new(move |u| sink.lock().unwrap().push(u.clone()));
    let options = TranscriptionOptions {
        language: Some("en".to_string()),
        beam_size: 0,
        progress: Some(cb),
        ..Default::default()
    };
    let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
    assert!(!segs.is_empty(), "no segments produced");

    let events = events.lock().unwrap();
    assert!(
        matches!(events.first(), Some(ProgressUpdate::Stage { kind: StageKind::Decode, .. })),
        "first event should be a Decode stage, got {events:?}"
    );
    for seg in &segs {
        assert!(
            events.iter().any(|p| matches!(p, ProgressUpdate::Segment(s) if s == seg)),
            "segment streamed for {seg:?}"
        );
    }
    assert!(
        matches!(events.last(), Some(ProgressUpdate::Stage { kind: StageKind::Done, fraction: Some(1.0), .. })),
        "last event should be Done with fraction 1.0, got {events:?}"
    );
}

#[test]
#[ignore = "network"]
fn cancel_before_run_returns_empty_and_emits_done() {
    let dev = NdArrayDevice::default();
    let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    let pcm = jfk_pcm();
    let events = Arc::new(Mutex::new(Vec::<ProgressUpdate>::new()));
    let sink = events.clone();
    let cb: Arc<dyn Fn(&ProgressUpdate) + Send + Sync> =
        Arc::new(move |u| sink.lock().unwrap().push(u.clone()));
    let cancelled = Arc::new(AtomicBool::new(true));
    let options = TranscriptionOptions {
        language: Some("en".to_string()),
        beam_size: 0,
        progress: Some(cb),
        cancelled,
        ..Default::default()
    };
    let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
    assert!(segs.is_empty(), "expected no segments, got {segs:?}");

    let events = events.lock().unwrap();
    assert!(
        !events.iter().any(|p| matches!(p, ProgressUpdate::Stage { kind: StageKind::Decode, .. })),
        "cancelled run must not start decoding, got {events:?}"
    );
    assert!(
        matches!(events.last(), Some(ProgressUpdate::Stage { kind: StageKind::Done, .. })),
        "cancelled run still emits Done, got {events:?}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test transcribe transcribe_streams_decode_segments_and_done_in_order -- --ignored`
Expected: FAIL to compile — `ProgressUpdate`, `StageKind`, `options.progress`,
`options.cancelled` do not exist.

- [ ] **Step 3: Implement**

In `crates/whisper-burn/src/transcribe.rs`:

Add imports:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
```

Add the public types above `TranscriptionOptions`:

```rust
/// Pipeline stages surfaced to UIs via the progress callback. `Download`
/// carries a completion `fraction` when the total size is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    Download,
    Weights,
    Audio,
    Decode,
    Done,
}

/// A live event from a running transcription pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgressUpdate {
    /// Stage transition with a human-readable `message`.
    Stage {
        kind: StageKind,
        message: String,
        fraction: Option<f32>,
    },
    /// One finished segment, streamed as it is decoded.
    Segment(TranscriptionSegment),
}

/// Shared progress sink; the desktop app forwards events to the UI.
pub type ProgressCallback = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;
```

Change `TranscriptionOptions` derive and add fields:

```rust
#[derive(Clone)]
pub struct TranscriptionOptions {
    // ... existing public fields unchanged ...
    /// Stream stage/segment events as the pipeline runs; `None` disables
    /// streaming.
    pub progress: Option<ProgressCallback>,
    /// Best-effort cancellation: set true and the decoder stops cleanly
    /// after the current window, returning segments produced so far.
    pub cancelled: Arc<AtomicBool>,
}
```

Add a manual `Debug` impl (the callback is not `Debug`):

```rust
impl std::fmt::Debug for TranscriptionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionOptions")
            .field("language", &self.language)
            .field("task", &self.task)
            .field("beam_size", &self.beam_size)
            .field("best_of", &self.best_of)
            .field("temperature", &self.temperature)
            .field("condition_on_previous_text", &self.condition_on_previous_text)
            .field("no_speech_threshold", &self.no_speech_threshold)
            .field("logprob_threshold", &self.logprob_threshold)
            .field("initial_prompt", &self.initial_prompt)
            .field("progress", &"<streaming callback>")
            .field("cancelled", &self.cancelled.load(Ordering::SeqCst))
            .finish()
    }
}
```

Add the two new fields to `Default`:

```rust
progress: None,
cancelled: Arc::new(AtomicBool::new(false)),
```

Add an emit helper and wire it into `transcribe`:

```rust
/// Forward a progress event to the configured callback (no-op when absent).
fn emit(options: &TranscriptionOptions, update: &ProgressUpdate) {
    if let Some(callback) = &options.progress {
        callback(update);
    }
}
```

In `transcribe`, the empty-chunk early return becomes:

```rust
if chunks.is_empty() {
    emit(
        options,
        &ProgressUpdate::Stage {
            kind: StageKind::Done,
            message: "no windows to decode".into(),
            fraction: None,
        },
    );
    return Ok(vec![]);
}
```

At the top of the window loop (right after `let seek_ms = ...`):

```rust
if options.cancelled.load(Ordering::SeqCst) {
    log::info!("decode: window {}/{}: cancelled", ci + 1, windows);
    break;
}
emit(
    options,
    &ProgressUpdate::Stage {
        kind: StageKind::Decode,
        message: format!("decode window {}/{}", ci + 1, windows),
        fraction: Some((ci + 1) as f32 / windows as f32),
    },
);
```

Replace the segment loop:

```rust
for segment in &segments[from..] {
    log::info!("  {}", segment_progress(segment));
    emit(options, &ProgressUpdate::Segment(segment.clone()));
}
```

Before the final `Ok(segments)`:

```rust
emit(
    options,
    &ProgressUpdate::Stage {
        kind: StageKind::Done,
        message: format!("{} segment(s)", segments.len()),
        fraction: Some(1.0),
    },
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test transcribe -- --ignored`
Expected: the two new tests PASS; the pre-existing network tests still pass.
Then `cargo test -p whisper-burn` (unit tests) — `Default`/`Debug` still compile.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn/src/transcribe.rs crates/whisper-burn/tests/transcribe.rs
git commit -m "feat: stream progress events and support cancellation in transcribe (whisper-burn)"
```

---

## T2: Download progress reporting

**User story:** As the app, I want model download progress bytes streamed so the
UI can show a download bar for large models (incl. the sharded ivrit merge).

**Files:**
- Modify: `crates/whisper-burn/src/download.rs`
- Modify: `crates/whisper-burn/src/model/whisper.rs`
- Test: `crates/whisper-burn/tests/download.rs`

**Interfaces:**
- Produces:
  - `pub type DownloadCallback = Arc<dyn Fn(u64, Option<u64>) + Send + Sync>`
  - `pub fn download_to_with_progress(url: &str, dest: &Path, overwrite: bool, on_progress: Option<DownloadCallback>) -> Result<PathBuf>`
  - `pub fn download_checkpoint_with_progress(size: ModelSize, dest_dir: &Path, overwrite: bool, require_weights: bool, on_progress: Option<DownloadCallback>) -> Result<PathBuf>`
  - `Whisper::<B>::from_pretrained_with_progress(size: ModelSize, device: B::Device, on_progress: Option<DownloadCallback>) -> Result<Self>` (gated `#[cfg(feature = "weights")]`)
- Consumes: existing `download_to`, `download_checkpoint` (become wrappers).

- [ ] **Step 1: Write the failing test**

Append to `crates/whisper-burn/tests/download.rs`:

```rust
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
```

Update the import line:

```rust
use whisper_burn::download::{cache_dir, download_checkpoint, download_to, download_to_with_progress, repo_base};
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test download download_to_with_progress_streams_bytes_and_total`
Expected: FAIL to compile — `download_to_with_progress` does not exist.

- [ ] **Step 3: Implement**

In `crates/whisper-burn/src/download.rs`:

```rust
pub type DownloadCallback = Arc<dyn Fn(u64, Option<u64>) + Send + Sync>;
```

Change `attempt_download` to take an optional callback and read the total from
`Content-Length`; copy incrementally:

```rust
fn attempt_download(url: &str, part: &Path, on_progress: &Option<DownloadCallback>) -> std::result::Result<(), DownloadFail> {
    let response = match ureq::get(url).call() {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            return Err(DownloadFail {
                kind: DownloadFailKind::Status,
                message: format!("HTTP {code} for {url} ({})", response.status_text()),
            });
        }
        Err(e) => {
            return Err(DownloadFail {
                kind: DownloadFailKind::Transport,
                message: e.to_string(),
            });
        }
    };
    let total = response
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut file = File::create(part).map_err(|e| DownloadFail {
        kind: DownloadFailKind::Transport,
        message: e.to_string(),
    })?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadFail {
            kind: DownloadFailKind::Transport,
            message: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| DownloadFail {
            kind: DownloadFailKind::Transport,
            message: e.to_string(),
        })?;
        downloaded += n as u64;
        if let Some(cb) = on_progress {
            cb(downloaded, total);
        }
    }
    file.flush().map_err(|e| DownloadFail {
        kind: DownloadFailKind::Transport,
        message: e.to_string(),
    })?;
    Ok(())
}
```

Add `use std::io::Read;` to the imports. Add the public functions:

```rust
/// Like [`download_to`] but reports progress through `on_progress`
/// (`downloaded` bytes, `total` when `Content-Length` is available).
pub fn download_to_with_progress(
    url: &str,
    dest: &Path,
    overwrite: bool,
    on_progress: Option<DownloadCallback>,
) -> Result<PathBuf> {
    if !overwrite && dest.exists() {
        return Ok(dest.to_path_buf());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(dest);
    match fetch_with_retry_on_progress(url, &part, &on_progress) {
        Ok(()) => {
            std::fs::rename(&part, dest)?;
            Ok(dest.to_path_buf())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

fn fetch_with_retry_on_progress(url: &str, part: &Path, on_progress: &Option<DownloadCallback>) -> Result<()> {
    for attempt in 0..MAX_ATTEMPTS {
        let last = match attempt_download(url, part, on_progress) {
            Ok(()) => return Ok(()),
            Err(fail) => {
                let _ = std::fs::remove_file(part);
                match fail.kind {
                    DownloadFailKind::Status => return Err(Error::Download(fail.message)),
                    DownloadFailKind::Transport => fail.message,
                }
            }
        };
        if attempt + 1 == MAX_ATTEMPTS {
            return Err(Error::Download(format!(
                "after {MAX_ATTEMPTS} attempts: {last}"
            )));
        }
    }
    unreachable!()
}

/// [`download_to`] with no progress reporting.
pub fn download_to(url: &str, dest: &Path, overwrite: bool) -> Result<PathBuf> {
    download_to_with_progress(url, dest, overwrite, None)
}
```

Keep `fetch_with_retry` (delete it only if unused after this change; `download_to`
now routes through the new retry helper). Replace the internals of
`download_checkpoint` to delegate, then add the progress variant:

```rust
pub fn download_checkpoint(
    size: ModelSize,
    dest_dir: &Path,
    overwrite: bool,
    require_weights: bool,
) -> Result<PathBuf> {
    download_checkpoint_with_progress(size, dest_dir, overwrite, require_weights, None)
}

pub fn download_checkpoint_with_progress(
    size: ModelSize,
    dest_dir: &Path,
    overwrite: bool,
    require_weights: bool,
    on_progress: Option<DownloadCallback>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let base = repo_base(size);
    log::info!("weights: fetching config.json from {base}");
    let config_path = download_to_with_progress(
        &format!("{base}/config.json"),
        &dest_dir.join("config.json"),
        overwrite,
        None,
    )?;
    if !require_weights {
        return Ok(config_path);
    }
    if size.is_sharded() {
        return download_sharded_checkpoint_with_progress(&base, dest_dir, overwrite, &on_progress);
    }
    log::info!("weights: fetching model.safetensors from {base}");
    let weights_path = download_to_with_progress(
        &format!("{base}/model.safetensors"),
        &dest_dir.join("model.safetensors"),
        overwrite,
        on_progress,
    )?;
    Ok(weights_path)
}
```

Thread the callback through `download_sharded_checkpoint` (rename to
`download_sharded_checkpoint_with_progress`, add an `on_progress:
&Option<DownloadCallback>` param) and pass `on_progress.clone()` into each
shard `download_to_with_progress` call. Existing single-file callers and the
shard merge behavior stay identical.

In `crates/whisper-burn/src/model/whisper.rs`, behind `#[cfg(feature = "weights")]`:

```rust
use crate::download::DownloadCallback;

pub fn from_pretrained_with_progress(
    size: ModelSize,
    device: B::Device,
    on_progress: Option<DownloadCallback>,
) -> Result<Self> {
    let dir = crate::download::cache_dir(size)?;
    if !dir.join("model.safetensors").exists() {
        log::info!("weights: {} not cached — fetching", size.repo_id());
        crate::download::download_checkpoint_with_progress(size, &dir, true, true, on_progress)?;
    } else {
        log::info!("weights: {} cached at {}", size.repo_id(), dir.display());
    }
    Self::load(size, &dir, device)
}
```

Make `from_pretrained` delegate to it with `None`. Export the alias: in
`crates/whisper-burn/src/download.rs` it is already `pub`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test download`
Expected: all download tests PASS (old + new).
Then the transfer test (run only if a live network test exists and is not
tagged `ignore`): `cargo test -p whisper-burn --test download downloads_tiny_checkpoint -- --ignored`.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn/src/download.rs crates/whisper-burn/src/model/whisper.rs crates/whisper-burn/tests/download.rs
git commit -m "feat: report download progress with content-length totals (whisper-burn)"
```

---

## T3: Shared turbo-translate validation

**User story:** As the app, I want the same turbo/translate protection the CLI
has, enforced in one library function the CLI also calls.

**Files:**
- Modify: `crates/whisper-burn/src/transcribe.rs`
- Modify: `crates/whisper-burn-cli/src/main.rs`
- Test: `crates/whisper-burn/tests/transcribe.rs`

**Interfaces:**
- Produces: `pub fn validate_model_task(model: ModelSize, task: Task) -> Result<()>`
  (exact error copy in Step 3 — the CLI test pins its text).

- [ ] **Step 1: Write the failing test**

Append to the non-network part of `crates/whisper-burn/tests/transcribe.rs`:

```rust
use whisper_burn::config::ModelSize;
use whisper_burn::transcribe::validate_model_task;

#[test]
fn turbo_translate_is_rejected_but_transcribe_and_large_v3_are_ok() {
    assert!(validate_model_task(ModelSize::LargeV3Turbo, Task::Translate).is_err());
    assert!(validate_model_task(ModelSize::LargeV3Turbo, Task::Transcribe).is_ok());
    assert!(validate_model_task(ModelSize::LargeV3, Task::Translate).is_ok());
    assert!(validate_model_task(ModelSize::IvritHebrew, Task::Translate).is_ok());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test transcribe turbo_translate_is_rejected_but_transcribe_and_large_v3_are_ok`
Expected: FAIL to compile — `validate_model_task` is not found.

- [ ] **Step 3: Implement**

In `crates/whisper-burn/src/transcribe.rs` add (above `validate_options`):

```rust
use crate::config::ModelSize;

/// Rejects model/task combinations the engine cannot honor: whisper
/// large-v3-turbo was fine-tuned on transcription data only and re-emits the
/// source language under `--task translate` (OpenAI model card).
pub fn validate_model_task(model: ModelSize, task: Task) -> Result<()> {
    if matches!(model, ModelSize::LargeV3Turbo) && matches!(task, Task::Translate) {
        return Err(Error::Unsupported(
            "whisper-large-v3-turbo was not trained for translation, \
             only transcription; --task translate would re-transcribe the \
             source language. Use --task transcribe, or translate with a \
             multilingual model (--model large-v3, medium, small, base, tiny)."
                .to_string(),
        ));
    }
    Ok(())
}
```

In `crates/whisper-burn-cli/src/main.rs` replace the inline turbo block:
delete the `if args.model == ModelSize::LargeV3Turbo && matches!(args.task, TaskArg::Translate) { ... }` block and the now-unused comment, and add after `validate_options(...)`:

```rust
use whisper_burn::transcribe::validate_model_task;
// ...
validate_model_task(args.model, args.task.into())?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test transcribe turbo_translate_is_rejected_but_transcribe_and_large_v3_are_ok`
Expected: PASS.
Run: `cargo test -p whisper-burn-cli --test cli turbo_translate_fails_fast_with_guidance`
Expected: PASS (the message text is unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn/src/transcribe.rs crates/whisper-burn/tests/transcribe.rs crates/whisper-burn-cli/src/main.rs
git commit -m "fix: share turbo-translate validation between CLI and app (whisper-burn)"
```

---

## T4: Extract shared `format` module into `whisper-burn`

**User story:** As the app, I want one copy of the txt/srt/vtt/json renderers
that the CLI already ships, so both can never drift.

**Files:**
- Create: `crates/whisper-burn/src/format.rs`
- Modify: `crates/whisper-burn/src/lib.rs`
- Modify: `crates/whisper-burn-cli/src/output.rs` (becomes a re-export)

**Interfaces:**
- Produces (all `pub`, identical signatures to today's CLI module):
  `first_strong_is_rtl(&str) -> bool`, `visual_order(&str) -> String`,
  `srt_time(u32) -> String`, `vtt_time(u32) -> String`,
  `bracket_line(&TranscriptionSegment) -> String`,
  `format_txt(&[TranscriptionSegment]) -> String`,
  `format_srt(&[TranscriptionSegment]) -> String`,
  `format_vtt(&[TranscriptionSegment]) -> String`,
  `format_json(&[TranscriptionSegment]) -> String`

- [ ] **Step 1: Write the failing test**

Create `crates/whisper-burn/tests/format.rs`:

```rust
use whisper_burn::format::{format_json, format_srt, format_txt, format_vtt};
use whisper_burn::TranscriptionSegment;

fn seg(start: u32, end: u32, text: &str) -> TranscriptionSegment {
    TranscriptionSegment { start, end, text: text.to_string() }
}

#[test]
fn txt_is_plain_logical_lines() {
    let out = format_txt(&[seg(0, 2000, "שלום עולם"), seg(2000, 5000, "hello")]);
    assert_eq!(out, "שלום עולם\nhello\n");
}

#[test]
fn json_serializes_segments() {
    let out = format_json(&[seg(0, 45000, "Hello")]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v[0]["start"], serde_json::json!(0.0));
    assert_eq!(v[0]["end"], serde_json::json!(45.0));
    assert_eq!(v[0]["text"], serde_json::json!("Hello"));
}

#[test]
fn srt_has_numbered_cues() {
    let out = format_srt(&[seg(1000, 2500, "Hi")]);
    assert!(out.starts_with("1\n00:00:01,000 --> 00:00:02,500\nHi\n"), "{out}");
}

#[test]
fn vtt_has_webvtt_header() {
    let out = format_vtt(&[seg(1000, 2500, "Hi")]);
    assert!(out.starts_with("WEBVTT\n"), "{out}");
    assert!(out.contains("00:01.000 --> 00:02.500"), "{out}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test format`
Expected: FAIL to compile — `whisper_burn::format` does not exist.

- [ ] **Step 3: Implement**

Copy the entire body of `crates/whisper-burn-cli/src/output.rs` into
`crates/whisper-burn/src/format.rs` untouched (it already imports only
`whisper_burn::TranscriptionSegment` and `serde_json`). Update the module doc
comment to mention the desktop app shares it.

In `crates/whisper-burn/src/lib.rs` add:

```rust
pub mod format;
```

Replace `crates/whisper-burn-cli/src/output.rs` with:

```rust
//! Transcript rendering for the CLI, shared with the desktop app via
//! `whisper_burn::format`.

pub use whisper_burn::format::{
    bracket_line, first_strong_is_rtl, format_json, format_srt, format_txt, format_vtt, srt_time,
    visual_order, vtt_time,
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test format`
Expected: PASS.
Run: `cargo test -p whisper-burn-cli --test output --test cli`
Expected: PASS (CLI output suite exercises the re-export).

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn/src/format.rs crates/whisper-burn/src/lib.rs crates/whisper-burn/tests/format.rs crates/whisper-burn-cli/src/output.rs
git commit -m "feat: move transcript format renderers into the library (whisper-burn)"
```

---

## T5: `.docx` and `.pdf` renderers

**User story:** As the app, I want Word and PDF exports generated in Rust so I
can save transcripts as files most users can open directly.

**Files:**
- Modify: `crates/whisper-burn/src/format.rs` (add `docx` and `pdf` inline submodules)
- Modify: `crates/whisper-burn/Cargo.toml` (deps `docx-rs`, `printpdf`; dev-dep `zip = "2"`)
- Modify: `Cargo.toml` (workspace deps `docx-rs = "0.3"`, `printpdf = "0.8"`, `zip = "2"`)
- Test: `crates/whisper-burn/tests/format.rs`

**Interfaces:**
- Produces:
  - `pub fn format_docx(segments: &[TranscriptionSegment], include_timestamps: bool) -> Result<Vec<u8>>`
  - `pub fn format_pdf(segments: &[TranscriptionSegment], include_timestamps: bool) -> Result<Vec<u8>>`
  - `pub(crate) fn segment_line(segment: &TranscriptionSegment, include_timestamps: bool) -> String`
  - `fn load_font(doc: &PdfDocumentReference) -> Result<IndirectFontRef>` (in `format::pdf`, private)

- [ ] **Step 1: Write the failing test**

Append to `crates/whisper-burn/tests/format.rs`:

```rust
use std::io::Read;
use whisper_burn::format::{format_docx, format_pdf};

#[test]
fn docx_embeds_segment_text_and_timestamps() {
    let segs = [seg(0, 2500, "שלום עולם")];
    let bytes = format_docx(&segs, true).unwrap();
    assert!(bytes.len() > 100, "unreasonably small docx");
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let mut xml = String::new();
    zip.by_name("word/document.xml").unwrap().read_to_string(&mut xml).unwrap();
    assert!(xml.contains("[00:00.000 --> 00:02.500]"), "timestamps missing:\n{xml}");
    assert!(xml.contains("שלום עולם"), "segment text missing:\n{xml}");
}

#[test]
fn pdf_produces_a_structural_pdf() {
    let segs = [seg(0, 2500, "שלום עולם")];
    let bytes = format_pdf(&segs, true).unwrap();
    assert!(bytes.starts_with(b"%PDF"), "not a PDF header");
    assert!(bytes.len() > 500, "unreasonably small pdf");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test format docx_embeds_segment_text_and_timestamps`
Expected: FAIL to compile — `format_docx` not found.

- [ ] **Step 3: Implement**

In `crates/whisper-burn/Cargo.toml`:

```toml
docx-rs.workspace = true
printpdf.workspace = true
```

In root `Cargo.toml`:

```toml
docx-rs = "0.3"
printpdf = "0.8"
zip = "2"
```

In `crates/whisper-burn/Cargo.toml` add dev-dependencies:

```toml
[dev-dependencies]
zip.workspace = true
serde_json.workspace = true
```

Append to `crates/whisper-burn/src/format.rs`:

```rust
/// Rendered line used by both the docx and pdf exporters: the segment text
/// with an optional leading `[start --> end]` bracket.
pub(crate) fn segment_line(segment: &TranscriptionSegment, include_timestamps: bool) -> String {
    if include_timestamps {
        format!(
            "[{} --> {}]  {}",
            vtt_time(segment.start),
            vtt_time(segment.end),
            segment.text
        )
    } else {
        segment.text.clone()
    }
}

pub mod docx {
    use crate::format::segment_line;
    use crate::{Error, Result, TranscriptionSegment};
    use docx_rs::{Docx, Paragraph, Run};

    /// One paragraph per segment. Word handles the surrounding text; for RTL
    /// paragraphs the paragraph is marked bidi so Hebrew lines up right.
    pub fn format_docx(
        segments: &[TranscriptionSegment],
        include_timestamps: bool,
    ) -> Result<Vec<u8>> {
        let mut doc = Docx::new();
        for segment in segments {
            let paragraph = Paragraph::new()
                .add_run(Run::new().add_text(segment_line(segment, include_timestamps)));
            doc = doc.add_paragraph(paragraph);
        }
        doc.build()
            .map_err(|e| Error::Unsupported(format!("docx build: {e}")))
    }
}

pub mod pdf {
    use crate::format::segment_line;
    use crate::{Error, Result, TranscriptionSegment};
    use printpdf::{BuiltinFont, IndirectFontRef, Mm, PdfDocument, PdfDocumentReference, Pt};

    const PAGE_W: f32 = 210.0; // A4 width, mm
    const PAGE_H: f32 = 297.0; // A4 height, mm
    const MARGIN: f32 = 20.0;  // mm
    const FONT_SIZE_PT: f32 = 10.0;
    const LINE_MM: f32 = 5.0;  // mm per line
    const CHARS_PER_LINE: usize = 66;

    /// A4 PDF with a title line and each segment wrapped to the page width.
    ///
    /// v1 layout is a simple char-based word wrap (fine for transcripts).
    pub fn format_pdf(
        segments: &[TranscriptionSegment],
        include_timestamps: bool,
    ) -> Result<Vec<u8>> {
        let (doc, page, layer) =
            PdfDocument::new("Transcriptor transcript", Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
        let current_layer = doc.get_page(page).get_layer(layer);
        let font = load_font(&doc)?;

        let mut y = Mm(PAGE_H - MARGIN);
        let text = if segments.is_empty() {
            "(no transcript)".to_string()
        } else {
            segments
                .iter()
                .map(|s| segment_line(s, include_timestamps))
                .collect::<Vec<_>>()
                .join("\n")
        };
        for paragraph in text.split('\n') {
            for line in wrap(paragraph, CHARS_PER_LINE) {
                if y.0 < MARGIN + LINE_MM {
                    let (_, page, layer) =
                        doc.add_page(Mm(PAGE_W), Mm(PAGE_H), "Layer 1");
                    y = Mm(PAGE_H - MARGIN);
                    current_layer_fmt(&doc, page, layer, &line, y, &font)?;
                } else {
                    current_layer_fmt(&doc, page, layer, &line, y, &font)?;
                }
                y = Mm(y.0 - LINE_MM);
            }
        }
        doc.save(&mut Vec::new()).map_err(|e| Error::Unsupported(format!("pdf build: {e}")))?;

        // Actually serialize: printpdf saves on drop from the reference; to keep
        // the API simple, build the byte buffer via serialize().
        serialize(&doc)
    }

    fn current_layer_fmt(
        doc: &PdfDocumentReference,
        page: printpdf::PageIndex,
        layer: printpdf::LayerIndex,
        line: &str,
        y: Mm,
        font: &IndirectFontRef,
    ) -> Result<()> {
        doc.get_page(page)
            .get_layer(layer)
            .use_text(line, FONT_SIZE_PT, Mm(MARGIN), y, font)
            .map_err(|e| Error::Unsupported(format!("pdf text: {e}")))
    }

    /// Break a paragraph at word boundaries, never exceeding `width` chars.
    fn wrap<'a>(paragraph: &'a str, width: usize) -> Vec<&'a str> {
        if paragraph.len() <= width {
            return vec![paragraph];
        }
        let mut lines = Vec::new();
        let mut start = 0usize;
        let bytes = paragraph.as_bytes();
        while start < bytes.len() {
            let end = (start + width).min(bytes.len());
            let mut cut = match bytes[start..end].iter().rposition(|&b| b == b' ') {
                Some(pos) if start + pos > start => Some(start + pos),
                _ => Some(end),
            };
            // avoid breaking inside multi-byte characters: backtrack to a char boundary
            if !paragraph.is_char_boundary(start + cut.unwrap_or(end)) {
                cut = Some(end);
            } else {
                // keep the space
            }
            let cut = cut.unwrap_or(end);
            lines.push(&paragraph[start..cut]);
            start = cut;
            // skip the separating space at the next line start
            if start < bytes.len() && bytes[start] == b' ' {
                start += 1;
            }
        }
        lines
    }

    fn serialize(doc: &PdfDocumentReference) -> Result<Vec<u8>> {
        use printpdf::PdfDocumentReference as _;
        let mut out = Vec::new();
        doc.save(&mut out).map_err(|e| Error::Unsupported(format!("pdf save: {e}")))?;
        Ok(out)
    }

    /// Unicode-capable system font, or None → built-in Helvetica fallback.
    /// The only platform-touching code in the app (candidate OS font paths);
    /// missing files are skipped so every OS still produces a PDF.
    fn load_font(doc: &PdfDocumentReference) -> Result<IndirectFontRef> {
        const CANDIDATES: &[&str] = &[
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ];
        for path in CANDIDATES {
            if let Ok(bytes) = std::fs::read(path) {
                return Ok(doc
                    .add_external_font(&bytes)
                    .map_err(|e| Error::Unsupported(format!("pdf font: {e}")))?);
            }
        }
        // ASCII-only fallback; still a valid PDF.
        Ok(doc
            .add_builtin_font(BuiltinFont::Helvetica)
            .map_err(|e| Error::Unsupported(format!("pdf font: {e}")))?)
    }
}
```

> **Executor note:** printpdf's exact save API changed across 0.7→0.8 (the
> `save(&mut Vec<u8>)` vs `PdfDocumentReference` drop). If `serialize` or the
> save call does not compile against the resolved 0.8.x, read the resolved
> crate's `PdfDocumentReference` docs and keep the byte-buffer output in
> `format_pdf`'s signature — the test only checks the result is a PDF.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test format`
Expected: PASS (docx zip/XML contains text + timestamps; pdf starts `%PDF`).
If the docx XML namespace wrapper changes the assertion, keep the assertion on
the `w:t` run text; verify with `zipinfo`-style inspection if needed.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/whisper-burn/Cargo.toml crates/whisper-burn/src/format.rs crates/whisper-burn/tests/format.rs
git commit -m "feat: add docx and pdf transcript renderers (whisper-burn)"
```

---

## T6: `ModelSize` parse/names/all in the library

**User story:** As the app, I want model lists and parse helpers in the library
so the CLI and the dropdown share one source of truth.

**Files:**
- Modify: `crates/whisper-burn/src/config.rs`
- Modify: `crates/whisper-burn-cli/src/main.rs`
- Test: `crates/whisper-burn/tests/config.rs`

**Interfaces:**
- Produces:
  - `pub const ModelSize::ALL: [ModelSize; 9] = [...]`
  - `pub fn ModelSize::cli_name(self) -> &'static str`
  - `pub fn ModelSize::parse(s: &str) -> Result<ModelSize>`

- [ ] **Step 1: Write the failing test**

Append to `crates/whisper-burn/tests/config.rs`:

```rust
#[test]
fn parse_and_cli_name_round_trip_for_every_model() {
    for model in ModelSize::ALL {
        assert_eq!(ModelSize::parse(model.cli_name()).unwrap(), model);
    }
    assert_eq!(ModelSize::parse("ivrit-hebrew").unwrap(), ModelSize::IvritHebrew);
    assert!(ModelSize::parse("nope").is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p whisper-burn --test config parse_and_cli_name_round_trip_for_every_model`
Expected: FAIL to compile — `ModelSize::ALL`, `cli_name`, `parse` missing.

- [ ] **Step 3: Implement**

In `crates/whisper-burn/src/config.rs` `impl ModelSize` add:

```rust
/// Every model, in CLI help order (also the dropdown order in the app).
pub const ALL: [ModelSize; 9] = [
    ModelSize::Tiny,
    ModelSize::Base,
    ModelSize::Small,
    ModelSize::Medium,
    ModelSize::Large,
    ModelSize::LargeV2,
    ModelSize::LargeV3,
    ModelSize::LargeV3Turbo,
    ModelSize::IvritHebrew,
];

/// The `--model`/dropdown key for this model.
pub fn cli_name(self) -> &'static str {
    match self {
        ModelSize::Tiny => "tiny",
        ModelSize::Base => "base",
        ModelSize::Small => "small",
        ModelSize::Medium => "medium",
        ModelSize::Large => "large",
        ModelSize::LargeV2 => "large-v2",
        ModelSize::LargeV3 => "large-v3",
        ModelSize::LargeV3Turbo => "large-v3-turbo",
        ModelSize::IvritHebrew => "ivrit-hebrew",
    }
}

/// Parse a CLI/dropdown model key.
pub fn parse(s: &str) -> Result<ModelSize> {
    Self::ALL
        .iter()
        .find(|m| m.cli_name() == s)
        .copied()
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "unknown model {s:?}; expected tiny, base, small, medium, large, \
                 large-v2, large-v3, large-v3-turbo, or ivrit-hebrew"
            ))
        })
}
```

In `crates/whisper-burn-cli/src/main.rs` replace `parse_model` with:

```rust
fn parse_model(s: &str) -> std::result::Result<ModelSize, String> {
    ModelSize::parse(s).map_err(|e| e.to_string())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p whisper-burn --test config parse_and_cli_name_round_trip_for_every_model`
Expected: PASS.
Run: `cargo test -p whisper-burn-cli --test cli cli_accepts_ivrit_hebrew_model`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn/src/config.rs crates/whisper-burn/tests/config.rs crates/whisper-burn-cli/src/main.rs
git commit -m "feat: expose ModelSize parse/names/all to the library (whisper-burn)"
```

---

## T7: Scaffold the Transcriptor Tauri app + React UI

**User story:** As a user, I want the app window to open with a working
frontend↔backend round trip, before any feature code exists.

**Files:**
- Create: `crates/transcriptor/src-tauri/Cargo.toml`, `build.rs`, `tauri.conf.json`, `capabilities/default.json`, `src/main.rs`, `src/lib.rs`
- Create: `crates/transcriptor/ui/package.json`, `package-lock.json` (via install), `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/index.css`, `src/vite-env.d.ts`
- Create: `crates/transcriptor/src-tauri/assets/icon.svg` + generated `icons/`
- Modify: root `Cargo.toml` (add member), root `.gitignore` (UI build dirs)

**Interfaces:**
- Produces:
  - Rust lib `transcriptor_lib::run()` that registers commands
    `get_defaults`, `inspect_audio`, `transcribe`, `stop`, `save_transcript`
    (stub bodies at this task; fully implemented in T8)
  - Frontend `invoke/get_defaults` round trip rendering a blank window

- [ ] **Step 1: Write the failing test**

The scaffold's test is that the React app calls `get_defaults` and renders the
count of models. Add to `crates/transcriptor/ui/src/App.tsx`:

```tsx
// (test hook — replaced by the real layout in T9/T10)
export function formatModelCount(input: unknown): string {
  const d = input as { models?: unknown[] };
  return `models: ${d.models?.length ?? 0}`;
}
```

The check for this task is manual but explicit:

- [ ] **Step 2: Create the Rust crate files**

`crates/transcriptor/src-tauri/Cargo.toml`:

```toml
[package]
name = "transcriptor"
version.workspace = true
edition.workspace = true

[lib]
name = "transcriptor_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
serde.workspace = true
serde_json.workspace = true
whisper-burn = { path = "../../whisper-burn", features = ["audio", "weights"] }
burn = { workspace = true, features = ["wgpu", "ndarray"] }
log.workspace = true
```

`crates/transcriptor/src-tauri/build.rs`:

```rust
fn main() {
    tauri_build::build()
}
```

`crates/transcriptor/src-tauri/src/main.rs`:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    transcriptor_lib::run();
}
```

`crates/transcriptor/src-tauri/src/lib.rs`:

```rust
mod app;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            app::get_defaults,
            app::inspect_audio,
            app::transcribe,
            app::stop,
            app::save_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

`crates/transcriptor/src-tauri/src/app.rs` (stubs — real logic in T8):

```rust
use serde::Serialize;

#[derive(Serialize)]
pub struct Defaults {
    pub models: Vec<String>,
    pub languages: Vec<String>,
    pub default_model: String,
    pub default_task: String,
    pub default_beam_size: usize,
    pub devices: Vec<String>,
}

#[tauri::command]
pub fn get_defaults() -> Result<Defaults, String> {
    let models = whisper_burn::config::ModelSize::ALL
        .iter()
        .map(|m| m.cli_name().to_string())
        .collect();
    let languages = {
        let mut v: Vec<String> = whisper_burn::decoding::lang::language_window(51866)
            .into_iter()
            .map(|(_, code)| code.to_string())
            .collect();
        v.insert(0, "auto".to_string());
        v
    };
    Ok(Defaults {
        models,
        languages,
        default_model: "tiny".to_string(),
        default_task: "transcribe".to_string(),
        default_beam_size: 0,
        devices: vec!["wgpu".to_string(), "cpu".to_string()],
    })
}

#[tauri::command]
pub fn inspect_audio(_path: String) -> Result<(), String> {
    Err("not implemented in scaffold".to_string())
}

#[tauri::command]
pub fn transcribe() -> Result<(), String> {
    Err("not implemented in scaffold".to_string())
}

#[tauri::command]
pub fn stop() -> Result<(), String> {
    Err("not implemented in scaffold".to_string())
}

#[tauri::command]
pub fn save_transcript() -> Result<(), String> {
    Err("not implemented in scaffold".to_string())
}
```

`crates/transcriptor/src-tauri/tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Transcriptor",
  "version": "0.1.0",
  "identifier": "com.whisperburn.transcriptor",
  "build": {
    "beforeDevCommand": "npm --prefix ../ui run dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "npm --prefix ../ui run build",
    "frontendDist": "../ui/dist"
  },
  "app": {
    "windows": [{ "title": "Transcriptor", "width": 1100, "height": 760 }],
    "security": { "csp": null }
  },
  "bundle": {
    "active": true,
    "targets": ["nsis", "msi"],
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.ico"
    ]
  }
}
```

`crates/transcriptor/src-tauri/capabilities/default.json`:

```json
{
  "identifier": "default",
  "windows": ["main"],
  "permissions": ["core:default", "dialog:default"]
}
```

- [ ] **Step 3: Create the icons**

`crates/transcriptor/src-tauri/assets/icon.svg` (a simple transcript glyph):

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="96" fill="#2563eb"/>
  <g fill="#ffffff" font-family="Segoe UI, sans-serif" font-weight="700" text-anchor="middle">
    <text x="256" y="170" font-size="64">&gt;&gt;</text>
    <text x="256" y="270" font-size="64">&gt;</text>
    <text x="256" y="370" font-size="44">Transcriptor</text>
  </g>
</svg>
```

Generate the icon set (this fills `src-tauri/icons/`):

```bash
npm --prefix crates/transcriptor/ui run tauri icon crates/transcriptor/src-tauri/assets/icon.svg
```

Verify `icons/icon.ico` exists under `src-tauri`.

- [ ] **Step 4: Create the frontend**

`crates/transcriptor/ui/package.json`:

```json
{
  "name": "transcriptor-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "tauri": "tauri"
  },
  "dependencies": {
    "@tauri-apps/api": "^2",
    "@tauri-apps/plugin-dialog": "^2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2",
    "@tailwindcss/vite": "^4",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "tailwindcss": "^4",
    "typescript": "^5.6.3",
    "vite": "^6"
  }
}
```

`crates/transcriptor/ui/vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
});
```

`crates/transcriptor/ui/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "strict": true,
    "skipLibCheck": true,
    "noEmit": true,
    "isolatedModules": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

`crates/transcriptor/ui/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Transcriptor</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`crates/transcriptor/ui/src/index.css`:

```css
@import "tailwindcss";
@custom-variant dark (&:where(.dark, .dark *));

@theme {
  --font-sans: system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

html, body, #root { height: 100%; }
body { @apply bg-slate-50 text-slate-900 antialiased; }
.dark body { @apply bg-slate-950 text-slate-100; }
```

`crates/transcriptor/ui/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

`crates/transcriptor/ui/src/App.tsx`:

```tsx
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface Defaults {
  models: string[];
  languages: string[];
  defaultModel: string;
  defaultTask: string;
  defaultBeamSize: number;
  devices: string[];
}

export function formatModelCount(input: unknown): string {
  const d = input as { models?: unknown[] };
  return `models: ${d.models?.length ?? 0}`;
}

export default function App() {
  const [count, setCount] = useState<string>("loading…");
  useEffect(() => {
    invoke<Defaults>("get_defaults")
      .then((d) => setCount(formatModelCount(d)))
      .catch((e) => setCount(`error: ${e}`));
  }, []);
  return (
    <div className="flex h-full items-center justify-center text-lg text-slate-500">
      Transcriptor scaffold — {count}
    </div>
  );
}
```

`crates/transcriptor/ui/src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
```

- [ ] **Step 5: Wire the workspace + install**

In root `Cargo.toml`, add the member:

```toml
members = ["crates/whisper-burn", "crates/whisper-burn-cli", "crates/transcriptor/src-tauri"]
```

Run:

```bash
npm --prefix crates/transcriptor/ui install
```

Commit `crates/transcriptor/ui/package-lock.json`.

Append to root `.gitignore`:

```gitignore
crates/transcriptor/ui/node_modules/
crates/transcriptor/ui/dist/
crates/transcriptor/src-tauri/target/
```

- [ ] **Step 6: Verify the window renders and the round trip works**

Run: `npm --prefix crates/transcriptor/ui run tauri dev`
Expected: the app window opens titled "Transcriptor" and shows
`Transcriptor scaffold — models: 9`.

Run: `cargo build -p transcriptor` from the workspace root
Expected: compiles cleanly (icons present, config valid). `cargo test
--workspace` must still pass.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml .gitignore crates/transcriptor
git commit -m "feat: scaffold Transcriptor Tauri app with React+Vite+Tailwind UI (transcriptor)"
```

---

## T8: App backend — state, errors, commands, events

**User story:** As the app, I want realistic commands that stream progress,
store the last transcript, and save typed files — fully tested without a
window.

**Files:**
- Create: `crates/transcriptor/src-tauri/src/error.rs`, `src/state.rs`,
  `src/core.rs`
- Modify: `crates/transcriptor/src-tauri/src/lib.rs`, `src/app.rs`
- Test: `crates/transcriptor/src-tauri/src/core.rs` (`#[cfg(test)]` unit tests)

**Interfaces:**
- Produces:
  - `pub struct AppError { pub kind: String, pub message: String }`
    (`Serialize`; `From<whisper_burn::Error>`)
  - `pub struct AppState { pub running: AtomicBool, pub cancel: Arc<AtomicBool>, pub last_segments: Mutex<Option<Vec<TranscriptionSegment>>> }`
  - `pub fn render(format: &str, include_timestamps: bool, segments: &[TranscriptionSegment]) -> Result<Vec<u8>, AppError>`
  - `pub struct TranscribeParams { model, language, task, beam_size, initial_prompt, device }` (`Deserialize`) with `fn into_options(self, cancel: Arc<AtomicBool>, progress: ProgressCallback) -> Result<TranscriptionOptions, AppError>`
  - `pub struct SegmentDto { start: u32, end: u32, text: String }` (`Serialize, Clone`)
  - `pub struct ProgressDto { kind: String, message: String, fraction: Option<f32> }` (`Serialize, Clone`)
  - `pub struct DoneDto { status: String, segments: Vec<SegmentDto>, message: Option<String> }` (`Serialize`)

- [ ] **Step 1: Write the failing tests**

Create `crates/transcriptor/src-tauri/src/core.rs` with the static logic and
failing unit tests appended:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use whisper_burn::format::{format_docx, format_json, format_pdf, format_srt, format_txt, format_vtt};
use whisper_burn::tokenizer::whisper::Task;
use whisper_burn::transcribe::{ProgressCallback, TranscriptionOptions, validate_model_task, validate_options};
use whisper_burn::{Error as WbError, ModelSize, Result as WbResult, TranscriptionSegment};

use crate::error::AppError;

// --- types -------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct TranscribeParams {
    pub model: String,
    pub language: Option<String>,
    pub task: String,
    pub beam_size: usize,
    pub initial_prompt: Option<String>,
    pub device: String,
}

#[derive(serde::Serialize, Clone)]
pub struct SegmentDto {
    pub start: u32,
    pub end: u32,
    pub text: String,
}

impl From<&TranscriptionSegment> for SegmentDto {
    fn from(s: &TranscriptionSegment) -> Self {
        Self { start: s.start, end: s.end, text: s.text.clone() }
    }
}

#[derive(serde::Serialize, Clone)]
pub struct ProgressDto {
    pub kind: String,
    pub message: String,
    pub fraction: Option<f32>,
}

#[derive(serde::Serialize)]
pub struct DoneDto {
    pub status: String,
    pub segments: Vec<SegmentDto>,
    pub message: Option<String>,
}

// --- pure logic --------------------------------------------------------------

pub fn parse_task(s: &str) -> Result<Task, AppError> {
    match s {
        "transcribe" => Ok(Task::Transcribe),
        "translate" => Ok(Task::Translate),
        other => Err(AppError::new("task", format!("unknown task {other:?}; expected transcribe or translate"))),
    }
}

pub fn render(
    format: &str,
    include_timestamps: bool,
    segments: &[TranscriptionSegment],
) -> Result<Vec<u8>, AppError> {
    match format {
        "txt" => Ok(format_txt(segments).into_bytes()),
        "srt" => Ok(format_srt(segments).into_bytes()),
        "vtt" => Ok(format_vtt(segments).into_bytes()),
        "json" => Ok(format_json(segments).into_bytes()),
        "docx" => format_docx(segments, include_timestamps).map_err(Into::into),
        "pdf" => format_pdf(segments, include_timestamps).map_err(Into::into),
        other => Err(AppError::new("format", format!("unsupported format {other:?}"))),
    }
}

pub fn validate_params(model: ModelSize, task: Task) -> Result<(), AppError> {
    validate_model_task(model, task).map_err(Into::into)
}

mod tests {
    use super::*;

    fn segs() -> Vec<TranscriptionSegment> {
        vec![
            TranscriptionSegment { start: 0, end: 2500, text: "שלום עולם".to_string() },
            TranscriptionSegment { start: 2500, end: 5000, text: "hello world".to_string() },
        ]
    }

    #[test]
    fn render_every_format_produces_bytes() {
        let segs = segs();
        for format in ["txt", "srt", "vtt", "json", "docx", "pdf"] {
            let out = render(format, true, &segs).unwrap_or_else(|e| panic!("{format}: {e}"));
            assert!(!out.is_empty(), "{format} empty");
        }
        let txt = render("txt", false, &segs).unwrap();
        assert!(String::from_utf8_lossy(&txt).contains("שלום עולם"));
    }

    #[test]
    fn parse_task_rejects_unknown() {
        assert_eq!(parse_task("transcribe").unwrap(), Task::Transcribe);
        assert_eq!(parse_task("translate").unwrap(), Task::Translate);
        assert!(parse_task("summarize").is_err());
    }

    #[test]
    fn turbo_translate_rejected_by_validate_params() {
        assert!(validate_params(ModelSize::LargeV3Turbo, Task::Translate).is_err());
        assert!(validate_params(ModelSize::LargeV3Turbo, Task::Transcribe).is_ok());
    }

    #[test]
    fn docx_render_includes_timestamp_brackets() {
        let out = render("docx", true, &segs()).unwrap();
        assert!(out.len() > 100);
    }
}
```

Create `crates/transcriptor/src-tauri/src/error.rs`:

```rust
#[derive(Debug, serde::Serialize)]
pub struct AppError {
    pub kind: String,
    pub message: String,
}

impl AppError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self { kind: kind.into(), message: message.into() }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AppError {}

impl From<whisper_burn::Error> for AppError {
    fn from(e: whisper_burn::Error) -> Self {
        use whisper_burn::Error as W;
        let kind = match &e {
            W::Io { .. } => "io",
            W::Download(_) => "download",
            W::MissingWeights { .. } => "missing-weights",
            W::UnsupportedFormat(_) => "format",
            W::ConfigParse(_) => "config",
            W::MissingWeight(_) => "weights",
            W::UnexpectedWeight(_) => "weights",
            W::ShapeMismatch { .. } => "shape",
            W::Decoder(_) => "decode",
            W::Tokenizer(_) => "tokenizer",
            W::AudioDecode(_) => "audio",
            W::Resample(_) => "audio",
            W::Unsupported(_) => "unsupported",
        };
        Self { kind: kind.to_string(), message: e.to_string() }
    }
}
```

> **Executor note:** verify the `whisper_burn::Error` variant names/field
> shapes against `crates/whisper-burn/src/error.rs`; adjust the match arms to
> the real variants (the variant set is listed in `tasks.md` T2).

Create `crates/transcriptor/src-tauri/src/state.rs`:

```rust
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use whisper_burn::TranscriptionSegment;

/// Process-shared state: guards one run at a time and keeps the last result
/// so save/copy commands can render without a live pipeline.
#[derive(Default)]
pub struct AppState {
    pub running: AtomicBool,
    pub cancel: Arc<AtomicBool>,
    pub last_segments: Mutex<Option<Vec<TranscriptionSegment>>>,
}

impl AppState {
    pub fn snapshot(&self) -> Vec<TranscriptionSegment> {
        self.last_segments.lock().unwrap().clone().unwrap_or_default()
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p transcriptor --lib core 2>&1`
Expected: FAIL to compile — `crate::error::AppError` and the new core items
do not exist yet (the test module references them). Realize the RED by
creating `error.rs`, `state.rs`, `core.rs` with the code above and running
until only assertion-level failures show.

- [ ] **Step 3: Implement the commands**

Replace `crates/transcriptor/src-tauri/src/app.rs` with the full
implementation, keep `get_defaults` from T7 verbatim, and add:

```rust
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use whisper_burn::audio::decode::decode_to_mono_f32;
use whisper_burn::backends::{BackendChoice, cpu_device, try_wgpu_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::transcribe::{ProgressCallback, ProgressUpdate, StageKind, transcribe};
use whisper_burn::{ModelSize, TranscriptionSegment};

use crate::core::{
    DoneDto, ProgressDto, SegmentDto, TranscribeParams, parse_task, render, validate_params,
};
use crate::error::AppError;
use crate::state::AppState;

// ... get_defaults (unchanged from T7) + Defaults struct ...

#[derive(serde::Serialize)]
pub struct AudioInfo {
    pub path: String,
    pub name: String,
    pub size_bytes: u64,
}

#[tauri::command]
pub fn inspect_audio(path: String) -> Result<AudioInfo, AppError> {
    let meta = std::fs::metadata(&path)
        .map_err(|e| AppError::new("io", format!("cannot read file: {e}")))?;
    if !meta.is_file() {
        return Err(AppError::new("io", "not a file"));
    }
    let name = Path::new(&path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.clone());
    Ok(AudioInfo { path, name, size_bytes: meta.len() })
}

#[tauri::command]
pub fn transcribe(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    path: String,
    params: TranscribeParams,
) -> Result<(), AppError> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err(AppError::new("busy", "a transcription is already running"));
    }
    state.cancel.store(false, Ordering::SeqCst);
    let state = Arc::clone(state.inner());
    std::thread::spawn(move || run_pipeline(app, &state, path, params));
    Ok(())
}

#[tauri::command]
pub fn stop(state: State<'_, Arc<AppState>>) -> Result<(), AppError> {
    state.cancel.store(true, Ordering::SeqCst);
    Ok(())
}

#[derive(serde::Deserialize)]
pub struct SaveRequest {
    pub path: String,
    pub format: String,
    pub include_timestamps: bool,
}

#[tauri::command]
pub fn save_transcript(
    state: State<'_, Arc<AppState>>,
    request: SaveRequest,
) -> Result<std::path::PathBuf, AppError> {
    let segments = state.snapshot();
    if segments.is_empty() {
        return Err(AppError::new("no-transcript", "nothing to save yet — transcribe first"));
    }
    let bytes = render(&request.format, request.include_timestamps, &segments)?;
    std::fs::write(&request.path, bytes)
        .map_err(|e| AppError::new("io", format!("cannot write file: {e}")))?;
    Ok(std::path::PathBuf::from(&request.path))
}

fn run_pipeline(app: AppHandle, state: &AppState, path: String, params: TranscribeParams) {
    let done = match core_run(&app, state, &path, &params) {
        Ok(segments) => {
            let cancelled = state.cancel.load(Ordering::SeqCst);
            DoneDto {
                status: if cancelled { "cancelled".into() } else { "finished".into() },
                segments: segments.iter().map(SegmentDto::from).collect(),
                message: if cancelled { Some("stopped by user".into()) } else { None },
            }
        }
        Err(e) => DoneDto {
            status: "error".into(),
            segments: vec![],
            message: Some(e.message.clone()),
        },
    };
    state.running.store(false, Ordering::SeqCst);
    let _ = app.emit("done", done);
}

fn core_run(
    app: &AppHandle,
    state: &AppState,
    path: &str,
    params: &TranscribeParams,
) -> std::result::Result<Vec<TranscriptionSegment>, AppError> {
    let model = ModelSize::parse(&params.model)?;
    let task = parse_task(&params.task)?;
    validate_params(model, task)?;

    let choice = match params.device.as_str() {
        "cpu" => BackendChoice::Cpu(cpu_device()),
        _ => match try_wgpu_device() {
            Some(device) => BackendChoice::Wgpu(device),
            None => BackendChoice::Cpu(cpu_device()),
        },
    };

    let progress = make_progress(app);
    let options = params
        .clone()
        .into_options(Arc::clone(&state.cancel), progress)?;

    emit_stage(app, "audio", format!("decoding {path}"), None);
    let (pcm, sample_rate) = decode_to_mono_f32(path)?;
    emit_stage(app, "audio", format!("audio: {:.1} s @ {} Hz", pcm.len() as f64 / sample_rate as f64, sample_rate), None);

    match choice {
        BackendChoice::Cpu(device) => run_backend::<burn::backend::ndarray::NdArray<f32>>(
            app, model, device, &pcm, sample_rate, &options,
        ),
        BackendChoice::Wgpu(device) => run_backend::<burn::backend::wgpu::Wgpu>(
            app, model, device, &pcm, sample_rate, &options,
        ),
    }
}

fn run_backend<B: burn::tensor::backend::Backend>(
    app: &AppHandle,
    model: ModelSize,
    device: &B::Device,
    pcm: &[f32],
    sample_rate: u32,
    options: &whisper_burn::TranscriptionOptions,
) -> std::result::Result<Vec<TranscriptionSegment>, AppError> {
    emit_stage(app, "weights", format!("loading {model}"), None);
    let whisper = Whisper::<B>::from_pretrained_with_progress(
        model,
        device.clone(),
        Some(make_download_callback(app)),
    )
    .map_err(AppError::from)?;
    emit_stage(app, "weights", format!("{model} ready"), None);
    transcribe(&whisper, pcm, sample_rate, options).map_err(AppError::from)
}

fn make_download_callback(app: &AppHandle) -> whisper_burn::download::DownloadCallback {
    let handle = app.clone();
    Arc::new(move |downloaded: u64, total: Option<u64>| {
        let fraction = total.map(|t| downloaded as f32 / t.max(1) as f32);
        let _ = handle.emit(
            "progress",
            ProgressDto {
                kind: "download".into(),
                message: format!("downloading checkpoint ({}/…)", downloaded / (1024 * 1024)),
                fraction,
            },
        );
    })
}

fn make_progress(app: &AppHandle) -> ProgressCallback {
    let handle = app.clone();
    Arc::new(move |update: &ProgressUpdate| match update {
        ProgressUpdate::Stage { kind, message, fraction } => {
            let kind = match kind {
                StageKind::Download => "download",
                StageKind::Weights => "weights",
                StageKind::Audio => "audio",
                StageKind::Decode => "decode",
                StageKind::Done => "done",
            };
            let _ = handle.emit(
                "progress",
                ProgressDto { kind: kind.to_string(), message: message.clone(), fraction: *fraction },
            );
        }
        ProgressUpdate::Segment(segment) => {
            let _ = handle.emit("segment", SegmentDto::from(segment));
        }
    })
}

fn emit_stage(app: &AppHandle, kind: &str, message: impl Into<String>, fraction: Option<f32>) {
    let _ = app.emit(
        "progress",
        ProgressDto { kind: kind.to_string(), message: message.into(), fraction },
    );
}
```

`State<'_, Arc<AppState>>` requires the app to manage an `Arc<AppState>`. Update
`lib.rs`:

```rust
mod app;
mod core;
mod error;
mod state;

use std::sync::Arc;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(state::AppState::default()))
        .invoke_handler(tauri::generate_handler![
            app::get_defaults,
            app::inspect_audio,
            app::transcribe,
            app::stop,
            app::save_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

In `core.rs`, finish `TranscribeParams::into_options`:

```rust
impl TranscribeParams {
    pub fn into_options(
        self,
        cancel: Arc<AtomicBool>,
        progress: ProgressCallback,
    ) -> std::result::Result<TranscriptionOptions, AppError> {
        let task = parse_task(&self.task)?;
        let model = ModelSize::parse(&self.model)?;
        validate_params(model, task)?;
        let language = self.language.filter(|l| l != "auto");
        let initial_prompt = self.initial_prompt.filter(|p| !p.trim().is_empty());
        let options = TranscriptionOptions {
            language,
            task,
            beam_size: self.beam_size,
            initial_prompt,
            progress: Some(progress),
            cancelled: cancel,
            ..Default::default()
        };
        validate_options(&options).map_err(Into::into)?;
        Ok(options)
    }
}
```

`core.rs` needs `#[derive(Clone)]` on `TranscribeParams` (used via `.clone()` in
`core_run`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p transcriptor --lib`
Expected: all core unit tests PASS.
Run: `cargo test -p whisper-burn -p whisper-burn-cli --offline 2>&1`
Expected: existing suites still green.

- [ ] **Step 5: Commit**

```bash
git add crates/transcriptor
git commit -m "feat: app backend commands, state, error mapping, and event streaming (transcriptor)"
```

---

## T9: UI shell, params rail, browse, dark mode

**User story:** As a user, I want the transcriptor layout with a form for the
whisper parameters, a file picker, and theme control.

**Files:**
- Create: `crates/transcriptor/ui/src/types.ts`
- Create: `crates/transcriptor/ui/src/components/ParamsRail.tsx`
- Create: `crates/transcriptor/ui/src/components/StatusPanel.tsx`
- Create: `crates/transcriptor/ui/src/components/TranscriptPanel.tsx`
- Modify: `crates/transcriptor/ui/src/App.tsx`
- Test: manual via `npm --prefix crates/transcriptor/ui run tauri dev`

**Interfaces:**
- Consumes (from T8/T7): `invoke("get_defaults")`, `invoke("inspect_audio", { path })`, `invoke("transcribe", { path, params })`, `invoke("stop")`, `invoke("save_transcript", { request })`, events `progress`/`segment`/`done`.

- [ ] **Step 1: Write the types + component with the failing data contract**

`crates/transcriptor/ui/src/types.ts`:

```ts
export interface Defaults {
  models: string[];
  languages: string[];
  defaultModel: string;
  defaultTask: string;
  defaultBeamSize: number;
  devices: string[];
}

export interface TaskParams {
  model: string;
  language: string;
  task: "transcribe" | "translate";
  beamSize: number;
  initialPrompt: string;
  device: string;
}

export interface ProgressDto {
  kind: string;
  message: string;
  fraction: number | null;
}

export interface SegmentDto {
  start: number;
  end: number;
  text: string;
}

export interface DoneDto {
  status: "finished" | "cancelled" | "error";
  segments: SegmentDto[];
  message: string | null;
}

export interface AudioInfo {
  path: string;
  name: string;
  sizeBytes: number;
}

export function clock(ms: number): string {
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(milli).padStart(3, "0")}`;
}
```

- [ ] **Step 2: Rewrite `App.tsx` with the shell wiring**

`crates/transcriptor/ui/src/App.tsx`:

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  AudioInfo,
  Defaults,
  DoneDto,
  ProgressDto,
  SegmentDto,
  TaskParams,
} from "./types";
import { ParamsRail } from "./components/ParamsRail";
import { StatusPanel } from "./components/StatusPanel";
import { TranscriptPanel } from "./components/TranscriptPanel";

const EXT: Record<string, string> = {
  txt: "txt", srt: "srt", vtt: "vtt", json: "json", docx: "docx", pdf: "pdf",
};

export default function App() {
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [params, setParams] = useState<TaskParams>({
    model: "tiny", language: "auto", task: "transcribe",
    beamSize: 0, initialPrompt: "", device: "wgpu",
  });
  const [audio, setAudio] = useState<AudioInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressDto[]>([]);
  const [segments, setSegments] = useState<SegmentDto[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  const segmentsRef = useRef<SegmentDto[]>([]);
  segmentsRef.current = segments;

  useEffect(() => {
    invoke<Defaults>("get_defaults")
      .then((d) => {
        setDefaults(d);
        setParams((p) => ({
          ...p, model: d.defaultModel, task: d.defaultTask as TaskParams["task"],
          beamSize: d.defaultBeamSize,
        }));
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    const p = listen<ProgressDto>("progress", (e) =>
      setProgress((prev) => [...prev.slice(-199), e.payload]),
    );
    const s = listen<SegmentDto>("segment", (e) =>
      setSegments((prev) => [...prev, e.payload]),
    );
    const d = listen<DoneDto>("done", (e) => {
      setRunning(false);
      if (e.payload.status === "error") {
        setError(e.payload.message);
        setSegments([]);
      } else {
        setError(null);
        setSegments(e.payload.segments);
      }
    });
    return () => {
      void p.then((f) => f());
      void s.then((f) => f());
      void d.then((f) => f());
    };
  }, []);

  const browse = useCallback(async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"] }],
    });
    if (typeof file !== "string") return;
    try {
      const info = await invoke<AudioInfo>("inspect_audio", { path: file });
      setAudio(info);
      setSegments([]);
      setProgress([]);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const run = useCallback(async () => {
    if (!audio || running) return;
    setError(null);
    setProgress([]);
    setSegments([]);
    setRunning(true);
    try {
      await invoke("transcribe", { path: audio.path, params });
    } catch (e) {
      setRunning(false);
      setError(String(e));
    }
  }, [audio, running, params]);

  const stop = useCallback(async () => {
    try { await invoke("stop"); } catch (e) { setError(String(e)); }
  }, []);

  const saveAs = useCallback(async (format: string, includeTimestamps: boolean) => {
    const target = await save({
      defaultPath: `transcript.${EXT[format] ?? format}`,
      filters: [{ name: format.toUpperCase(), extensions: [EXT[format] ?? format] }],
    });
    if (!target) return;
    try {
      await invoke("save_transcript", {
        request: { path: target, format, includeTimestamps },
      });
    } catch (e) {
      setError(String(e));
    }
  }, []);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3 dark:border-slate-800 dark:bg-slate-900">
        <div>
          <h1 className="text-lg font-semibold">Transcriptor</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Whisper on burn — transcribe or translate audio
          </p>
        </div>
        <button
          onClick={() => setDark((v) => !v)}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          {dark ? "Light" : "Dark"}
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <ParamsRail
          defaults={defaults}
          params={params}
          onChange={setParams}
          audio={audio}
          running={running}
          onBrowse={browse}
          onRun={run}
          onStop={stop}
        />
        <main className="flex min-h-0 flex-1 flex-col gap-3 p-4">
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
              {error}
            </div>
          )}
          <StatusPanel progress={progress} />
          <TranscriptPanel segments={segments} running={running} onSave={saveAs} />
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create the components**

`crates/transcriptor/ui/src/components/ParamsRail.tsx`:

```tsx
import type { AudioInfo, Defaults, TaskParams } from "../types";

interface Props {
  defaults: Defaults | null;
  params: TaskParams;
  onChange: (p: TaskParams) => void;
  audio: AudioInfo | null;
  running: boolean;
  onBrowse: () => void;
  onRun: () => void;
  onStop: () => void;
}

export function ParamsRail(props: Props) {
  const { defaults, params, onChange, audio, running, onBrowse, onRun, onStop } = props;
  const label = "block text-sm font-medium text-slate-600 dark:text-slate-300";
  const select =
    "mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100";
  return (
    <aside className="w-72 shrink-0 space-y-4 overflow-y-auto border-r border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div>
        <span className={label}>Audio file</span>
        <button
          onClick={onBrowse}
          className="mt-1 w-full rounded-lg border border-slate-300 px-2.5 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          {audio ? audio.name : "Browse…"}
        </button>
        {audio && (
          <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">
            {(audio.sizeBytes / (1024 * 1024)).toFixed(1)} MB
          </p>
        )}
      </div>

      <div>
        <label className={label}>Model</label>
        <select
          className={select}
          value={params.model}
          disabled={running}
          onChange={(e) => onChange({ ...params, model: e.target.value })}
        >
          {(defaults?.models ?? []).map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div>
        <label className={label}>Language</label>
        <select
          className={select}
          value={params.language}
          disabled={running}
          onChange={(e) => onChange({ ...params, language: e.target.value })}
        >
          {(defaults?.languages ?? []).map((l) => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>
      </div>

      <div>
        <label className={label}>Task</label>
        <div className="mt-1 flex gap-3 text-sm">
          {(["transcribe", "translate"] as const).map((t) => (
            <label key={t} className="flex items-center gap-1.5">
              <input
                type="radio"
                name="task"
                checked={params.task === t}
                disabled={running}
                onChange={() => onChange({ ...params, task: t })}
              />
              {t}
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className={label}>Beam size (0 = greedy)</label>
        <input
          type="number"
          min={0}
          className={select}
          value={params.beamSize}
          disabled={running}
          onChange={(e) => onChange({ ...params, beamSize: Number(e.target.value) })}
        />
      </div>

      <div>
        <label className={label}>Initial prompt</label>
        <input
          value={params.initialPrompt}
          disabled={running}
          onChange={(e) => onChange({ ...params, initialPrompt: e.target.value })}
          placeholder="Optional prompt for the first window"
          className={select}
        />
      </div>

      <div>
        <label className={label}>Device</label>
        <select
          className={select}
          value={params.device}
          disabled={running}
          onChange={(e) => onChange({ ...params, device: e.target.value })}
        >
          {(defaults?.devices ?? ["wgpu", "cpu"]).map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      <div className="flex gap-2 pt-1">
        <button
          onClick={onRun}
          disabled={running || !audio}
          className="flex-1 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Working…" : "Transcribe"}
        </button>
        <button
          onClick={onStop}
          disabled={!running}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          Stop
        </button>
      </div>
    </aside>
  );
}
```

`crates/transcriptor/ui/src/components/StatusPanel.tsx`:

```tsx
import type { ProgressDto } from "../types";

export function StatusPanel({ progress }: { progress: ProgressDto[] }) {
  const lastDownload = [...progress].reverse().find((p) => p.kind === "download");
  const fraction =
    lastDownload && lastDownload.fraction != null ? lastDownload.fraction : null;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h2 className="mb-2 text-sm font-semibold text-slate-500 dark:text-slate-400">
        Status
      </h2>
      {progress.length === 0 && (
        <p className="text-sm text-slate-400">Pick an audio file and press Transcribe.</p>
      )}
      {fraction != null && (
        <div className="mb-3 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
          <div
            className="h-full bg-blue-600 transition-all"
            style={{ width: `${Math.round(fraction * 100)}%` }}
          />
        </div>
      )}
      <ul className="space-y-1 text-sm">
        {progress.slice(-50).map((p, i) => (
          <li key={i} className="flex gap-2 text-slate-600 dark:text-slate-300">
            <span className="shrink-0 rounded bg-slate-100 px-1.5 text-xs uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
              {p.kind}
            </span>
            <span className="truncate font-mono">{p.message}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
```

`crates/transcriptor/ui/src/components/TranscriptPanel.tsx`:

```tsx
import { useCallback, useState } from "react";
import { clock, type SegmentDto } from "../types";

function renderText(segments: SegmentDto[], timestamps: boolean): string {
  return segments
    .map((s) =>
      timestamps ? `[${clock(s.start)} --> ${clock(s.end)}]  ${s.text}` : s.text,
    )
    .join("\n");
}

interface Props {
  segments: SegmentDto[];
  running: boolean;
  onSave: (format: string, timestamps: boolean) => void;
}

export function TranscriptPanel({ segments, running, onSave }: Props) {
  const [timestamps, setTimestamps] = useState(true);
  const [format, setFormat] = useState("txt");

  const copy = useCallback(async () => {
    const text = renderText(segments, timestamps);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  }, [segments, timestamps]);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400">
          Transcript {running && segments.length > 0 ? `(${segments.length} segments…)` : ""}
        </h2>
        <div className="flex items-center gap-2 text-sm">
          <label className="flex items-center gap-1 text-slate-600 dark:text-slate-300">
            <input
              type="checkbox"
              checked={timestamps}
              onChange={(e) => setTimestamps(e.target.checked)}
            />
            timestamps
          </label>
          <button
            onClick={copy}
            disabled={segments.length === 0}
            className="rounded-lg border border-slate-300 px-2.5 py-1 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Copy
          </button>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
          >
            {["txt", "srt", "vtt", "json", "docx", "pdf"].map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
          <button
            onClick={() => onSave(format, timestamps)}
            disabled={segments.length === 0}
            className="rounded-lg bg-blue-600 px-3 py-1 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
      <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap p-4 font-mono text-sm leading-relaxed">
        {segments.length === 0
          ? running
            ? "Transcribing…"
            : "Transcript will appear here."
          : renderText(segments, timestamps)}
      </pre>
    </section>
  );
}
```

- [ ] **Step 4: Verify in the running app**

Run: `npm --prefix crates/transcriptor/ui run tauri dev`
Manual checks:
- Defaults render: model default "tiny", language "auto", beam 0, wgpu.
- Browse picks a `.wav`, name + size shown in the rail.
- Dark/light toggles the whole window.

- [ ] **Step 5: Commit**

```bash
git add crates/transcriptor/ui
git commit -m "feat: params rail, status panel, transcript panel, theme toggle UI (transcriptor)"
```

> **Bug-gate note (use superpowers:systematic-debugging):** this task's manual
> UI gives the first real chance to find event-ordering or param-shape bugs.
> Do not "fix" the UI silently; reproduce, then change the backend contract in
> T8-reviewed code and its tests.

---

## T10: Streaming run/stop wiring (progress + segments live)

**User story:** As a user, I want to press Transcribe and watch the stages and
segments stream in, and to stop mid-way keeping partial output.

**Files:**
- Modify: `crates/transcriptor/ui/src/App.tsx` (already wired in T9 — verify only)
- Test: manual + the T8 Rust suite

**Interfaces:**
- Consumes: `invoke("transcribe")` events `progress`/`segment`/`done`.

- [ ] **Step 1: Verify live streaming with the small default model**

Run: `npm --prefix crates/transcriptor/ui run tauri dev`
Pick `test_16000_mono.wav` (English/Hebrew fixture), keep model `tiny`:
- Status shows `audio`, `weights`, `decode` stage chips as they happen.
- Segment lines appear in the transcript panel while running.
- On completion the panel freezes the final text; `running` clears.

- [ ] **Step 2: Verify Stop keeps partial output**

Start a run on `test_16000_mono.wav` with model `large-v3`, press **Stop**
mid-run. Expected: status shows decode stages, then a `done` event with
`"cancelled"`; any segments produced remain visible.

- [ ] **Step 3: Verify error surfacing**

Point Browse at a non-audio file (e.g. `tasks2.md` copy) and Transcribe.
Expected: `error:` status row from `AppError` mapping; no crash; app usable.

- [ ] **Step 4: Regression gate**

Run: `cargo test --workspace` and `cargo clippy --workspace --all-targets`
(clippy warnings: fix in the touched code). Expect green.

- [ ] **Step 5: Commit**

```bash
git add crates/transcriptor
git commit -m "feat: stream transcription progress and segments into the UI (transcriptor)"
```

---

## T11: Save formats, copy, and export polish

**User story:** As a user, I want to save the transcript as txt/srt/vtt/json/
docx/pdf with or without timestamps, and copy to the clipboard.

**Files:**
- Modify: `crates/transcriptor/ui/src/components/TranscriptPanel.tsx` (already
  implemented in T9 — the target of this task's verification)
- Test: `crates/transcriptor/src-tauri/src/core.rs` (extend `render` tests)

**Interfaces:**
- Consumes: `invoke("save_transcript", { request })`, library `format_*`.

- [ ] **Step 1: Extend render tests in `core.rs`**

Append inside `mod tests`:

```rust
#[test]
fn render_formats_match_extension_content_rules() {
    let segs = segs();
    let txt = String::from_utf8(render("txt", true, &segs).unwrap()).unwrap();
    assert!(txt.contains("שלום עולם"));
    let json = String::from_utf8(render("json", false, &segs).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v.len(), 2);
    assert!(render("docx", true, &segs).unwrap().len() > 100);
    assert!(render("pdf", true, &segs).unwrap().starts_with(b"%PDF"));
    assert!(render("nope", true, &segs).is_err());
}
```

- [ ] **Step 2: Run the tests (RED first, then GREEN)**

Run: `cargo test -p transcriptor --lib`
Expected: new test passes immediately (render already handles these formats
from T8) — that is fine for an integration lock; the RED for `render` itself
was T8. This step locks the behavior.

- [ ] **Step 3: Verify every save format end-to-end**

In `tauri dev`, run a short transcription (tiny, English), then save each of
`txt`, `srt`, `vtt`, `json`, `docx`, `pdf` via the Save menu (with and without
timestamps). Verify on disk:
- txt/srt/vtt/json decode as UTF-8 with the expected layout
- docx opens in Word/LibreOffice with the text
- pdf opens with the Hebrew/English text rendered (exercises the
  system-font picker; note the fallback if no Unicode font is found)
- Copy pastes the timestamped text into a text editor

- [ ] **Step 4: Commit**

```bash
git add crates/transcriptor
git commit -m "feat: verified save/copy for all six transcript formats (transcriptor)"
```

---

## T12: Windows packaging + smoke test + manual E2E

**User story:** As a user, I want a Windows installer produced from the app I
used, and a first smoke test that the installed app works.

**Files:**
- Modify: `crates/transcriptor/src-tauri/tauri.conf.json` (no changes expected)
- Verify: `crates/transcriptor/ui/src-tauri/target/release/bundle/nsis/*.exe`, `.../bundle/msi/*.msi`

- [ ] **Step 1: Build installers**

From the workspace root:

```bash
npm --prefix crates/transcriptor/ui run tauri build
```

Expected: `target/release/bundle/nsis/Transcriptor_0.1.0_x64-setup.exe` and
`target/release/bundle/msi/Transcriptor_0.1.0_x64_en-US.msi` are produced
(the path is under `crates/transcriptor/src-tauri/target/...`).

- [ ] **Step 2: Smoke test the installer**

Install the MSI on Windows, launch "Transcriptor", and confirm: window opens,
defaults load (models:9), a tiny transcription of `test_16000_mono.wav`
completes, one save round-trips (pdf). Uninstall cleanly.

- [ ] **Step 3: Manual E2E regression pass**

Repeat the spec's manual checklist against the built app:
- English clip (jfk.flac) transcribes correctly
- Hebrew clip (`test_16000_mono.wav`) renders logical-order in the panel
- mp3 (`test_411000_s.mp3`) decodes
- translate task returns English for the Hebrew clip with a multilingual model
- run with `ivrit-hebrew` model (downloads 2 shards, merged, progress bar
  shows download fraction)
- cancel mid-run keeps partial output
- dark mode toggle + every save format

- [ ] **Step 4: Commit any fixes as follow-up commits**

Any fixes found in Step 2/3 land as normal TDD commits with their own tests
(reproduce-first rule). No code is committed without its test unless the fix
is UI-presentational and verified manually.

- [ ] **Step 5: Commit the packaging milestone**

```bash
git add -A crates/transcriptor
git commit -m "feat: Windows NSIS+MSI packaging and installer smoke test (transcriptor)"
```

---

## Self-review notes (fixed inline during writing)

- `TranscriptionOptions` keeps `Default`/`Debug` (custom `Debug`; `Default`
  gains `progress: None`, `cancelled: AtomicBool`).
- Event payload shapes in T8 (`ProgressDto`/`SegmentDto`/`DoneDto`) match the
  TS types in T9; `SegmentDto` is the wire format everywhere (the library
  `TranscriptionSegment` never crosses the IPC boundary).
- `ModelSize::ALL` order (T6) matches CLI help order and the T7 dropdown.
- `validate_model_task` (T3) is the single source of the turbo rule; T8 calls
  it in both `validate_params` and `into_options` (call once per run, in
  `into_options`).
- The `language_window(51866)` call (T7/T8) uses the large-v3 vocab count per
  the Global Constraints.