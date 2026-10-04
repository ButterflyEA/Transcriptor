# Dictation Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Work one sprint at a time**; do not start a sprint until the previous sprint's exit criteria pass and the user okays it.

**Goal:** A background push-to-talk dictation service: hold a global hotkey, speak, release, and the transcript is pasted into the focused field and left on the clipboard, using `whisper-burn` locally.

**Architecture:** One Rust binary `dictation` in a new workspace crate. A pure state machine (`controller`) is driven by inputs from pluggable edges (hotkey backends, control socket, capture, engine) and emits actions executed by a thin runtime. Platform-specific code (hotkey, paste, tray) sits behind small traits so the core is unit-tested without hardware.

**Tech Stack:** Rust 2024, `whisper-burn` (feature `audio` only; no network), `cpal`, `arboard`, `enigo`, `rdev`, `interprocess`, `tray-icon`, `eframe`/`egui`, `serde` + `toml`, `dirs`, `clap`, `log`.

**Spec:** [docs/superpowers/specs/2026-10-04-dictation-service-design.md](../specs/2026-10-04-dictation-service-design.md)

## Global Constraints

- Platforms: Windows and Linux (X11 and Wayland, incl. Hyprland/Omarchy via compositor binds).
- Push-to-talk: hold to record, release to transcribe. Default hotkey `ctrl+space`; chord release = release of either key.
- Output: ALWAYS paste into the focused field AND leave the text on the clipboard (never restore the old clipboard).
- Press during transcription: start recording immediately; queue the audio FIFO, queue cap 3 (config `queue_cap`), transcribe and paste in order.
- Recordings shorter than 300 ms are discarded. Max recording 120 s (config `max_record_secs`).
- Language: config `language` is `"auto"` or a lowercase 2-letter code. Default model is set by the Sprint 1 spike (expected `ivrit-turbo`); default language stays `"auto"`.
- Everything local: **no network access at all** (airgapped). The service never downloads anything and does not enable the `whisper-burn` `weights` feature. The installer bundles the model.
- Bundled model: `ivrit-ai/whisper-large-v3-turbo` (Hebrew, Apache-2.0, single `model.safetensors`), loaded with `ModelSize::LargeV3Turbo` dimensions via `Whisper::load(size, &dir, device)`. Must run on a **4 GB VRAM** GPU; the exact precision (f32 / f16) or fallback model is decided by the Sprint 1 spike.
- Autostart, toggle mode, VAD, streaming, macOS, per-app paste profiles are OUT of scope.
- Edition 2024; workspace dependency versions live in the root `Cargo.toml`.
- Commit trailer on every commit: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Corrections to the spec discovered while planning

These are folded into Sprint 0, Task 0.1:

1. `whisper_burn::transcribe(&Whisper<B>, pcm: &[f32], sample_rate: u32, &TranscriptionOptions) -> Result<TranscriptionResult>` **already takes in-memory PCM and resamples internally** (`audio` feature). No `whisper-burn` change is needed, and `capture` only downmixes to mono and reports its native sample rate (no resampling in `capture`).
2. The settings window is a **separate process** (`dictation settings`, eframe) launched from the tray, because `tray-icon` and `eframe` each want to own the main-thread event loop. It talks to the service via the control socket (`reload`).

## File Structure

```
crates/dictation-service/
  Cargo.toml
  src/
    main.rs            # clap: run (default) | start | stop | toggle | status | reload | settings
    lib.rs             # module declarations
    config.rs          # Config, load/save/validate, default_path
    chord.rs           # Chord parsing + ChordTracker (pure)
    controller.rs      # state machine: Input -> Vec<Action> (pure)
    audio.rs           # downmix + Recording type (pure) + cpal Recorder
    engine.rs          # model load + transcribe_pcm
    output.rs          # clipboard + paste backends
    control.rs         # Command parse/format (pure) + IPC server/client
    hotkey/mod.rs      # HotkeyBackend trait + select()
    hotkey/rdev_backend.rs
    hotkey/evdev_backend.rs   # cfg(linux), feature "evdev"
    runtime.rs         # wires channels, threads, executes Actions
    tray.rs            # tray icon + status + menu
    settings.rs        # eframe settings app (own process)
  tests/
    config.rs chord.rs controller.rs audio.rs control.rs engine.rs
```

---

# Sprint 0 — Foundation (crate, spec fixes, config)

**Exit criteria:** `cargo test -p dictation-service` passes; `dictation --help` runs; config round-trips and rejects bad values.

### Task 0.1: Scaffold crate and correct the spec

**Files:**
- Modify: `Cargo.toml` (workspace members)
- Create: `crates/dictation-service/Cargo.toml`, `src/lib.rs`, `src/main.rs`
- Modify: `docs/superpowers/specs/2026-10-04-dictation-service-design.md`

**Produces:** crate `dictation-service`, lib name `dictation_service`, binary `dictation`.

- [ ] **Step 1: Add the workspace member**

In `Cargo.toml` change members to:
```toml
members = ["crates/whisper-burn", "crates/whisper-burn-cli", "crates/transcriptor/src-tauri", "crates/dictation-service"]
```
Add to `[workspace.dependencies]`: `toml = "0.8"`, `dirs = "6"`.

- [ ] **Step 2: Create `crates/dictation-service/Cargo.toml`**

```toml
[package]
name = "dictation-service"
version.workspace = true
edition.workspace = true

[lib]
name = "dictation_service"

[[bin]]
name = "dictation"
path = "src/main.rs"

[features]
default = []
evdev = ["dep:evdev"]

[dependencies]
whisper-burn = { path = "../whisper-burn", features = ["audio"] }
burn = { workspace = true, features = ["wgpu", "ndarray"] }
serde.workspace = true
toml.workspace = true
dirs.workspace = true
clap.workspace = true
log.workspace = true
thiserror.workspace = true

[target.'cfg(target_os = "linux")'.dependencies]
evdev = { version = "0.13", optional = true }
```

- [ ] **Step 3: Create `src/lib.rs` and `src/main.rs`**

`src/lib.rs`: leave it empty in this task (modules are added as they are created).
`src/main.rs`:
```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "dictation", version, about = "Push-to-talk dictation with whisper-burn")]
struct Args {}

fn main() {
    let _ = Args::parse();
}
```

- [ ] **Step 4: Edit the spec**

In the spec: replace the engine bullet's "Requires a public entry point ... part of this work." with "Uses the existing `whisper_burn::transcribe` (takes in-memory PCM and a sample rate; resamples internally). No `whisper-burn` change needed."; in **capture** replace "downmixes to mono and resamples to 16 kHz (reusing `rubato`)" with "downmixes to mono and reports its native sample rate (whisper-burn resamples)"; in **ui** state "settings window runs as a separate `dictation settings` process (eframe) because tray and egui each need the main-thread event loop; it signals the service with `reload`".

- [ ] **Step 5: Verify build**

Run: `cargo build -p dictation-service`
Expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/dictation-service docs/superpowers/specs
git commit -m "feat(dictation): scaffold crate and correct spec" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 0.2: Config

**Files:**
- Create: `crates/dictation-service/src/config.rs`, `crates/dictation-service/tests/config.rs`
- Modify: `crates/dictation-service/src/lib.rs`

**Produces:**
```rust
pub struct Config { pub hotkey: String, pub model: String, pub model_dir: Option<PathBuf>, pub language: String,
  pub input_device: Option<String>, pub paste_delay_ms: u64,
  pub max_record_secs: u32, pub queue_cap: usize }
impl Default for Config
impl Config { pub fn validate(&self) -> Result<(), ConfigError>;
  pub fn load_from(path: &Path) -> Result<Config, ConfigError>;   // missing file => default
  pub fn save_to(&self, path: &Path) -> Result<(), ConfigError>;
  pub fn language_option(&self) -> Option<String>; }               // "auto" => None
pub fn default_path() -> PathBuf
pub enum ConfigError { Io(String), Parse(String), Invalid(String) }
```

- [ ] **Step 1: Write failing tests** in `tests/config.rs`

```rust
use dictation_service::config::Config;

#[test]
fn defaults_are_valid_and_match_spec() {
    let c = Config::default();
    assert_eq!(c.hotkey, "ctrl+space");
    assert_eq!(c.model, "ivrit-turbo");
    assert_eq!(c.model_dir, None);
    assert_eq!(c.language, "auto");
    assert_eq!(c.queue_cap, 3);
    assert_eq!(c.max_record_secs, 120);
    assert!(c.validate().is_ok());
    assert_eq!(c.language_option(), None);
}

#[test]
fn round_trips_through_disk() {
    let dir = std::env::temp_dir().join(format!("dict-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut c = Config::default();
    c.language = "he".into();
    c.save_to(&path).unwrap();
    assert_eq!(Config::load_from(&path).unwrap(), c);
    assert_eq!(c.language_option(), Some("he".to_string()));
}

#[test]
fn missing_file_yields_default() {
    let p = std::env::temp_dir().join("dict-definitely-missing/config.toml");
    assert_eq!(Config::load_from(&p).unwrap(), Config::default());
}

#[test]
fn rejects_bad_values() {
    for mutate in [
        (|c: &mut Config| c.model = "huge".into()) as fn(&mut Config),
        |c| c.language = "english".into(),
        |c| c.queue_cap = 0,
        |c| c.max_record_secs = 0,
        |c| c.hotkey = "".into(),
    ] {
        let mut c = Config::default();
        mutate(&mut c);
        assert!(c.validate().is_err());
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test -p dictation-service --test config`
Expected: FAIL (unresolved import `dictation_service::config`).

- [ ] **Step 3: Implement `src/config.rs`**

```rust
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use whisper_burn::ModelSize;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// Models the service can run. `IvritTurbo` is the bundled Hebrew model
/// (ivrit-ai/whisper-large-v3-turbo); it shares `LargeV3Turbo` dimensions.
pub fn parse_model(name: &str) -> Option<ModelSize> {
    match name {
        "ivrit-turbo" => Some(ModelSize::LargeV3Turbo),
        other => ModelSize::parse(other).ok(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hotkey: String,
    pub model: String,
    pub model_dir: Option<PathBuf>,
    pub language: String,
    pub input_device: Option<String>,
    pub paste_delay_ms: u64,
    pub max_record_secs: u32,
    pub queue_cap: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "ctrl+space".into(),
            model: "ivrit-turbo".into(),
            model_dir: None,
            language: "auto".into(),
            input_device: None,
            paste_delay_ms: 50,
            max_record_secs: 120,
            queue_cap: 3,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| Err(ConfigError::Invalid(m.into()));
        if self.hotkey.trim().is_empty() {
            return bad("hotkey is empty");
        }
        if parse_model(&self.model).is_none() {
            return bad("unknown model");
        }
        let lang_ok = self.language == "auto"
            || (self.language.len() == 2 && self.language.chars().all(|c| c.is_ascii_lowercase()));
        if !lang_ok {
            return bad("language must be \"auto\" or a 2-letter lowercase code");
        }
        if self.queue_cap == 0 {
            return bad("queue_cap must be >= 1");
        }
        if self.max_record_secs == 0 {
            return bad("max_record_secs must be >= 1");
        }
        Ok(())
    }

    pub fn language_option(&self) -> Option<String> {
        (self.language != "auto").then(|| self.language.clone())
    }

    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(ConfigError::Io(e.to_string())),
        };
        let cfg: Self = toml::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        self.validate()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| ConfigError::Io(e.to_string()))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| ConfigError::Parse(e.to_string()))?;
        std::fs::write(path, text).map_err(|e| ConfigError::Io(e.to_string()))
    }
}

pub fn default_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dictation")
        .join("config.toml")
}
```
Add `pub mod config;` to `lib.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p dictation-service --test config`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dictation-service
git commit -m "feat(dictation): config load/save/validate" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

# Sprint 1 — Engine and 4 GB VRAM spike (offline Hebrew model)

**Exit criteria:** `dictation transcribe-file` transcribes a Hebrew clip from a local model folder with no network; engine tests pass; the VRAM decision (Task 1.3) is made and recorded by the user. **Sprints 2+ do not start before this gate.**

### Task 1.1: Engine

**Files:**
- Create: `crates/dictation-service/src/engine.rs`, `tests/engine.rs`
- Modify: `src/lib.rs`

**Consumes (verified in `whisper-burn`):** `whisper_burn::model::whisper::Whisper::<B>::load(size: ModelSize, weights_dir: &Path, device: B::Device) -> Result<Whisper<B>>` (offline; expects `config.json` + `model.safetensors` in the dir), `whisper_burn::backends::{BackendChoice, default_device}`, `whisper_burn::{transcribe, TranscriptionOptions, TranscriptionSegment}` where `Segment { start: u32, end: u32, text: String }`.

**Produces:**
```rust
pub fn join_segments(segments: &[TranscriptionSegment]) -> String
pub fn resolve_model_dir(configured: Option<&Path>, exe_dir: &Path, data_dir: &Path) -> Option<PathBuf> // first of: configured, exe_dir/models/ivrit-turbo, data_dir/models/ivrit-turbo that contains model.safetensors
pub struct Engine { /* private */ }
impl Engine {
  pub fn load(size: ModelSize, model_dir: &Path) -> whisper_burn::Result<Engine>;
  pub fn transcribe_pcm(&self, pcm: &[f32], sample_rate: u32, language: Option<&str>) -> whisper_burn::Result<String>;
}
```

- [ ] **Step 1: Failing unit test for `join_segments`** (in `tests/engine.rs`)

```rust
use dictation_service::engine::join_segments;
use whisper_burn::TranscriptionSegment;

fn seg(t: &str) -> TranscriptionSegment {
    TranscriptionSegment { start: 0, end: 1000, text: t.into() }
}

#[test]
fn joins_trims_and_collapses_whitespace() {
    let s = [seg(" Hello"), seg("  world. "), seg("")];
    assert_eq!(join_segments(&s), "Hello world.");
}

#[test]
fn empty_is_empty() {
    assert_eq!(join_segments(&[]), "");
}
```

- [ ] **Step 2: Run, expect FAIL** — `cargo test -p dictation-service --test engine` (unresolved `engine`).

- [ ] **Step 3: Implement `src/engine.rs`**

```rust
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::Wgpu;
use whisper_burn::backends::{BackendChoice, default_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::{ModelSize, Result, TranscriptionOptions, TranscriptionSegment, transcribe};
use std::path::{Path, PathBuf};

enum Loaded {
    Cpu(Whisper<NdArray<f32>>),
    Wgpu(Whisper<Wgpu>),
}

pub struct Engine {
    model: Loaded,
}

/// Segments are joined with single spaces; whisper segments carry leading spaces.
pub fn join_segments(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .flat_map(|s| s.text.split_whitespace())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn resolve_model_dir(configured: Option<&Path>, exe_dir: &Path, data_dir: &Path) -> Option<PathBuf> {
    let candidates = configured
        .map(Path::to_path_buf)
        .into_iter()
        .chain([exe_dir.join("models/ivrit-turbo"), data_dir.join("models/ivrit-turbo")]);
    candidates.into_iter().find(|d| d.join("model.safetensors").is_file())
}

impl Engine {
    pub fn load(size: ModelSize, model_dir: &Path) -> Result<Self> {
        let model = match default_device() {
            BackendChoice::Cpu(d) => Loaded::Cpu(Whisper::load(size, model_dir, d)?),
            BackendChoice::Wgpu(d) => Loaded::Wgpu(Whisper::load(size, model_dir, d)?),
        };
        Ok(Self { model })
    }

    pub fn transcribe_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let options = TranscriptionOptions {
            language: language.map(str::to_owned),
            ..Default::default()
        };
        let result = match &self.model {
            Loaded::Cpu(m) => transcribe(m, pcm, sample_rate, &options)?,
            Loaded::Wgpu(m) => transcribe(m, pcm, sample_rate, &options)?,
        };
        Ok(join_segments(&result.segments))
    }
}
```
Add `pub mod engine;` to `lib.rs`. If `TranscriptionSegment` is not re-exported at the paths above, use the full module paths seen in `crates/whisper-burn/src/lib.rs`.

- [ ] **Step 4: Run unit tests** — `cargo test -p dictation-service --test engine`; Expected: 2 passed.

- [ ] **Step 5: Add `resolve_model_dir` tests** (temp dirs): configured dir wins when it has `model.safetensors`; falls back to `exe_dir/models/ivrit-turbo`; returns `None` when no candidate has the file (this is the "missing model, no download" error path). Run and expect pass.

- [ ] **Step 6: Add ignored integration test** (needs the model in `DICTATION_MODEL_DIR`)

```rust
#[test]
#[ignore = "needs the model files in DICTATION_MODEL_DIR"]
fn transcribes_silence_without_error() {
    let dir = std::path::PathBuf::from(std::env::var("DICTATION_MODEL_DIR").unwrap());
    let engine = dictation_service::engine::Engine::load(whisper_burn::ModelSize::LargeV3Turbo, &dir).unwrap();
    let pcm = vec![0.0f32; 16_000 * 2];
    let text = engine.transcribe_pcm(&pcm, 16_000, Some("he")).unwrap();
    assert!(text.len() < 40, "silence should not hallucinate a paragraph: {text:?}");
}
```
Run: `cargo test -p dictation-service --test engine -- --ignored`; Expected: pass.

- [ ] **Step 7: Commit** — `feat(dictation): offline engine over whisper-burn` (with trailer).

### Task 1.2: Dev subcommand `transcribe-file` (warm-up and latency check)

**Files:** Modify `src/main.rs`.

Add a hidden subcommand that decodes a file with `whisper_burn::audio::decode::decode_to_mono_f32`, loads the model from a directory, runs `Engine::transcribe_pcm`, and prints text plus `load_ms=` and `elapsed_ms=`.

- [ ] **Step 1:** Replace `Args` with a clap `Subcommand` enum containing `#[command(hide = true)] TranscribeFile { path: PathBuf, #[arg(long)] model_dir: PathBuf, #[arg(long, default_value = "ivrit-turbo")] model: String, #[arg(long)] language: Option<String> }`; resolve `model` with `config::parse_model`.
- [ ] **Step 2:** Run `cargo run --release -p dictation-service -- transcribe-file <some.wav> --model-dir <dir> --language he`. Expected: transcript + timings.
- [ ] **Step 3:** Record load and transcribe latency for a 5 s clip in the PR description.
- [ ] **Step 4:** Commit `feat(dictation): transcribe-file dev command`.

### Task 1.3: SPIKE — fit the Hebrew model in 4 GB VRAM (decision gate)

**Output is a recommendation, not shipped code.** Anything built here is throwaway unless the user approves keeping it.

**Question:** Does `ivrit-ai/whisper-large-v3-turbo` load and transcribe a 30 s Hebrew clip within 4 GB VRAM, and in what precision?

- [ ] **Step 1: Obtain the model on a connected machine** (one-time): download `config.json` and `model.safetensors` from `https://huggingface.co/ivrit-ai/whisper-large-v3-turbo` into a local folder (`<dir>`). Record the sha256 of `model.safetensors` for the installer manifest.
- [ ] **Step 2: Baseline f32.** Run `transcribe-file` on a 5 s and a 30 s Hebrew clip with `--language he`. While it runs, record peak GPU memory (Windows: Task Manager "Dedicated GPU memory" or `nvidia-smi --query-gpu=memory.used --format=csv -l 1`; Linux: `nvidia-smi`/`radeontop`) and peak process RAM. Record whether load succeeds on a GPU limited to 4 GB (the target card, or force a cap) and whether output is correct Hebrew.
- [ ] **Step 2b: Check dimensions.** Confirm `Whisper::load` accepts the checkpoint with `ModelSize::LargeV3Turbo` (it validates dims); if it rejects it, note the exact error — a new `ModelSize` variant in `whisper-burn` becomes a prerequisite task.
- [ ] **Step 3: If f32 does not fit with headroom (peak > ~3.6 GB), try f16.** Check whether `burn::backend::wgpu::Wgpu<half::f16>` (or burn's f16 element type) is usable on the target adapter in a throwaway branch of `whisper-burn`: make `Whisper<B>` generic use f16 elements, load the F16-converted weights, compare the transcript to the f32 output on the same clips (character diff) and re-measure VRAM.
- [ ] **Step 4: Write up** results as a table (precision, peak VRAM, peak RAM, load s, 5 s latency, 30 s latency, transcript diff vs f32) in the PR description.
- [ ] **Step 5: Decision gate — STOP and ask the user** to pick one of: (a) ship f32 ivrit turbo; (b) add f16 compute to `whisper-burn` (becomes a new task in this sprint, with golden tests unaffected for f32); (c) fall back to a smaller model (e.g. a Hebrew fine-tune of `whisper-small`, which needs weight-format and dims checks). Record the choice in the spec ("Model handling") and set `Config::default().model` accordingly.
- [ ] **Step 6:** Commit the spec update `docs(dictation): record 4GB VRAM decision`.

---

# Sprint 2 — Capture (microphone to PCM)

**Exit criteria:** `dictation record-test --secs 3` records from the default mic and prints sample count, rate, peak level; pure downmix tests pass.

### Task 2.1: Recording type and downmix (pure)

**Files:** Create `src/audio.rs`, `tests/audio.rs`; modify `src/lib.rs`.

**Produces:**
```rust
pub struct Recording { pub samples: Vec<f32>, pub sample_rate: u32 }
impl Recording { pub fn duration_ms(&self) -> u64 }
pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32>
pub const MIN_RECORDING_MS: u64 = 300;
```

- [ ] **Step 1: Failing tests** (`tests/audio.rs`)

```rust
use dictation_service::audio::{Recording, downmix_to_mono};

#[test]
fn mono_passthrough() {
    assert_eq!(downmix_to_mono(&[0.1, 0.2], 1), vec![0.1, 0.2]);
}

#[test]
fn stereo_is_averaged() {
    assert_eq!(downmix_to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
}

#[test]
fn trailing_partial_frame_is_dropped() {
    assert_eq!(downmix_to_mono(&[1.0, 1.0, 9.0], 2), vec![1.0]);
}

#[test]
fn duration_ms_from_rate() {
    let r = Recording { samples: vec![0.0; 8000], sample_rate: 16_000 };
    assert_eq!(r.duration_ms(), 500);
}
```
- [ ] **Step 2:** Run `cargo test -p dictation-service --test audio`; expect FAIL.
- [ ] **Step 3: Implement**

```rust
pub const MIN_RECORDING_MS: u64 = 300;

#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Recording {
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 { return 0; }
        self.samples.len() as u64 * 1000 / self.sample_rate as u64
    }
}

pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let n = channels.max(1) as usize;
    interleaved
        .chunks_exact(n)
        .map(|f| f.iter().sum::<f32>() / n as f32)
        .collect()
}
```
- [ ] **Step 4:** Run tests; expect 4 passed.
- [ ] **Step 5:** Commit `feat(dictation): recording type and downmix`.

### Task 2.2: cpal Recorder

**Files:** Modify `Cargo.toml` (`cpal = "0.15"`), `src/audio.rs`, `src/main.rs`.

**Produces:**
```rust
pub trait Recorder: Send {
    fn start(&mut self) -> Result<(), String>;
    fn stop(&mut self) -> Result<Recording, String>;   // after start(); drains the buffer
}
pub struct CpalRecorder;
impl CpalRecorder { pub fn new(device_name: Option<String>, max_secs: u32) -> Self }
```
`cpal::Stream` is `!Send`, so `CpalRecorder` runs a dedicated thread owning the stream and exchanging `Start`/`Stop(reply Sender)` messages over `std::sync::mpsc`. Samples go into an `Arc<Mutex<Vec<f32>>>` capped at `max_secs * rate` (extra samples dropped). Handle `SampleFormat::{F32,I16,U16}` by converting to f32 before `downmix_to_mono`.

- [ ] **Step 1:** Implement the thread + trait. Device selection: by exact name if set and found, otherwise default input device; return `Err("no input device")` when none.
- [ ] **Step 2:** Add hidden `record-test --secs N` subcommand: start, sleep N, stop, print `samples`, `rate`, `peak`.
- [ ] **Step 3:** Manual check: run it and speak. Expected: peak > 0.05; with mic muted peak ≈ 0.
- [ ] **Step 4:** Add a unit test for the cap using a helper `push_capped(buf: &mut Vec<f32>, chunk: &[f32], max: usize)` (extracted so it is testable): `push_capped(&mut v, &[1.0;5], 3)` leaves `v.len() == 3`.
- [ ] **Step 5:** Commit `feat(dictation): cpal recorder`.

---

# Sprint 3 — Core loop (state machine, output, runtime) — first end-to-end

**Exit criteria:** controller tests pass; `dictation run` plus a temporary dev trigger (stdin: `s`=press, `e`=release) records, transcribes, copies to the clipboard and pastes into the focused field.

### Task 3.1: Controller state machine (pure)

**Files:** Create `src/controller.rs`, `tests/controller.rs`; modify `src/lib.rs`.

**Produces:**
```rust
pub enum Input { Press, Release, Captured(Recording), CaptureFailed(String),
                 Transcribed(Result<String, String>), EngineReady, EngineFailed(String) }
pub enum Action { StartCapture, StopCapture, Transcribe(Recording),
                  Paste(String), SetStatus(Status), Notify(String) }
pub enum Status { Loading, Idle, Recording, Transcribing, Error }
pub struct Controller { /* private */ }
impl Controller { pub fn new(queue_cap: usize) -> Self;
                  pub fn handle(&mut self, input: Input) -> Vec<Action>; }
```
Rules: nothing is recorded or transcribed until `EngineReady`... except presses while `Loading` are ignored with `Notify("model still loading")`. `Press` while recording is ignored; `Release` while not recording is ignored. `Captured` shorter than `MIN_RECORDING_MS` is discarded silently. If the engine is busy the recording is queued; if the queue already holds `queue_cap` jobs, drop it and `Notify("queue full")`. On `Transcribed(Ok(text))` with empty text: no paste. Status priority: Recording > Transcribing > Idle.

- [ ] **Step 1: Failing tests** (`tests/controller.rs`)

```rust
use dictation_service::audio::Recording;
use dictation_service::controller::{Action, Controller, Input, Status};

fn rec(ms: u64) -> Recording {
    Recording { samples: vec![0.1; (16 * ms) as usize], sample_rate: 16_000 }
}
fn ready() -> Controller {
    let mut c = Controller::new(3);
    c.handle(Input::EngineReady);
    c
}
fn has(a: &[Action], f: impl Fn(&Action) -> bool) -> bool { a.iter().any(f) }

#[test]
fn press_is_ignored_while_loading() {
    let mut c = Controller::new(3);
    let a = c.handle(Input::Press);
    assert!(!has(&a, |x| matches!(x, Action::StartCapture)));
    assert!(has(&a, |x| matches!(x, Action::Notify(_))));
}

#[test]
fn press_release_transcribe_paste_flow() {
    let mut c = ready();
    assert!(has(&c.handle(Input::Press), |x| matches!(x, Action::StartCapture)));
    assert!(has(&c.handle(Input::Release), |x| matches!(x, Action::StopCapture)));
    let a = c.handle(Input::Captured(rec(1000)));
    assert!(has(&a, |x| matches!(x, Action::Transcribe(_))));
    let a = c.handle(Input::Transcribed(Ok("hello".into())));
    assert!(has(&a, |x| matches!(x, Action::Paste(t) if t == "hello")));
    assert!(has(&a, |x| matches!(x, Action::SetStatus(Status::Idle))));
}

#[test]
fn short_recordings_are_discarded() {
    let mut c = ready();
    c.handle(Input::Press);
    c.handle(Input::Release);
    let a = c.handle(Input::Captured(rec(100)));
    assert!(!has(&a, |x| matches!(x, Action::Transcribe(_))));
}

#[test]
fn press_while_transcribing_queues_in_order() {
    let mut c = ready();
    c.handle(Input::Press); c.handle(Input::Release);
    c.handle(Input::Captured(rec(1000)));            // job 1 starts
    assert!(has(&c.handle(Input::Press), |x| matches!(x, Action::StartCapture)));
    c.handle(Input::Release);
    let a = c.handle(Input::Captured(rec(2000)));    // queued, engine busy
    assert!(!has(&a, |x| matches!(x, Action::Transcribe(_))));
    let a = c.handle(Input::Transcribed(Ok("one".into())));
    assert!(has(&a, |x| matches!(x, Action::Paste(t) if t == "one")));
    assert!(has(&a, |x| matches!(x, Action::Transcribe(r) if r.duration_ms() == 2000)));
}

#[test]
fn queue_is_capped() {
    let mut c = Controller::new(1);
    c.handle(Input::EngineReady);
    c.handle(Input::Press); c.handle(Input::Release); c.handle(Input::Captured(rec(1000))); // running
    c.handle(Input::Press); c.handle(Input::Release); c.handle(Input::Captured(rec(1000))); // queued (1/1)
    c.handle(Input::Press); c.handle(Input::Release);
    let a = c.handle(Input::Captured(rec(1000)));                                            // dropped
    assert!(has(&a, |x| matches!(x, Action::Notify(m) if m.contains("queue"))));
}

#[test]
fn empty_transcript_is_not_pasted() {
    let mut c = ready();
    c.handle(Input::Press); c.handle(Input::Release); c.handle(Input::Captured(rec(1000)));
    let a = c.handle(Input::Transcribed(Ok("   ".into())));
    assert!(!has(&a, |x| matches!(x, Action::Paste(_))));
}

#[test]
fn transcription_failure_notifies_and_moves_on() {
    let mut c = ready();
    c.handle(Input::Press); c.handle(Input::Release); c.handle(Input::Captured(rec(1000)));
    let a = c.handle(Input::Transcribed(Err("boom".into())));
    assert!(has(&a, |x| matches!(x, Action::Notify(_))));
}

#[test]
fn double_press_and_stray_release_are_ignored() {
    let mut c = ready();
    c.handle(Input::Press);
    assert!(c.handle(Input::Press).is_empty());
    c.handle(Input::Release);
    assert!(c.handle(Input::Release).is_empty());
}
```
- [ ] **Step 2:** Run `cargo test -p dictation-service --test controller`; expect FAIL.
- [ ] **Step 3: Implement `src/controller.rs`**

```rust
use std::collections::VecDeque;

use crate::audio::{MIN_RECORDING_MS, Recording};

#[derive(Debug)]
pub enum Input {
    Press,
    Release,
    Captured(Recording),
    CaptureFailed(String),
    Transcribed(Result<String, String>),
    EngineReady,
    EngineFailed(String),
}

#[derive(Debug)]
pub enum Action {
    StartCapture,
    StopCapture,
    Transcribe(Recording),
    Paste(String),
    SetStatus(Status),
    Notify(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Loading,
    Idle,
    Recording,
    Transcribing,
    Error,
}

pub struct Controller {
    ready: bool,
    recording: bool,
    busy: bool,
    queue: VecDeque<Recording>,
    queue_cap: usize,
}

impl Controller {
    pub fn new(queue_cap: usize) -> Self {
        Self { ready: false, recording: false, busy: false, queue: VecDeque::new(), queue_cap }
    }

    fn status(&self) -> Status {
        if !self.ready {
            Status::Loading
        } else if self.recording {
            Status::Recording
        } else if self.busy {
            Status::Transcribing
        } else {
            Status::Idle
        }
    }

    fn start_next(&mut self, out: &mut Vec<Action>) {
        if self.busy {
            return;
        }
        if let Some(job) = self.queue.pop_front() {
            self.busy = true;
            out.push(Action::Transcribe(job));
        }
    }

    pub fn handle(&mut self, input: Input) -> Vec<Action> {
        let mut out = Vec::new();
        match input {
            Input::EngineReady => self.ready = true,
            Input::EngineFailed(m) => {
                out.push(Action::SetStatus(Status::Error));
                out.push(Action::Notify(format!("model failed to load: {m}")));
                return out;
            }
            Input::Press if !self.ready => {
                return vec![Action::Notify("model still loading".into())];
            }
            Input::Press if self.recording => return Vec::new(),
            Input::Press => {
                self.recording = true;
                out.push(Action::StartCapture);
            }
            Input::Release if !self.recording => return Vec::new(),
            Input::Release => out.push(Action::StopCapture),
            Input::CaptureFailed(m) => {
                self.recording = false;
                out.push(Action::Notify(format!("recording failed: {m}")));
            }
            Input::Captured(rec) => {
                self.recording = false;
                if rec.duration_ms() >= MIN_RECORDING_MS {
                    if self.queue.len() >= self.queue_cap {
                        out.push(Action::Notify("transcription queue full; recording dropped".into()));
                    } else {
                        self.queue.push_back(rec);
                        self.start_next(&mut out);
                    }
                }
            }
            Input::Transcribed(result) => {
                self.busy = false;
                match result {
                    Ok(text) if !text.trim().is_empty() => out.push(Action::Paste(text)),
                    Ok(_) => {}
                    Err(m) => out.push(Action::Notify(format!("transcription failed: {m}"))),
                }
                self.start_next(&mut out);
            }
        }
        out.push(Action::SetStatus(self.status()));
        out
    }
}
```
Note: `EngineFailed` leaves `ready == false`, so later presses get "model still loading"; Task 7.1 refines this so the Error status persists.

- [ ] **Step 4:** Run tests until all 8 pass.
- [ ] **Step 5:** Commit `feat(dictation): controller state machine`.

### Task 3.2: Output (clipboard + paste)

**Files:** Modify `Cargo.toml` (`arboard = "3"`, `enigo = "0.3"`); create `src/output.rs`; modify `src/lib.rs`.

**Produces:**
```rust
pub trait Paster: Send { fn paste(&mut self, text: &str) -> Result<PasteOutcome, String>; }
pub enum PasteOutcome { Pasted, ClipboardOnly(String /* reason */) }
pub fn system_paster(paste_delay_ms: u64) -> Box<dyn Paster>
pub fn is_wayland() -> bool       // XDG_SESSION_TYPE == "wayland" || WAYLAND_DISPLAY set
pub fn pick_paste_tool(has: impl Fn(&str) -> bool) -> Option<&'static str>   // "wtype" then "ydotool"
```

- [ ] **Step 1: Failing test** for `pick_paste_tool` in `tests/output.rs`:
```rust
use dictation_service::output::pick_paste_tool;
#[test] fn prefers_wtype() { assert_eq!(pick_paste_tool(|_| true), Some("wtype")); }
#[test] fn falls_back_to_ydotool() { assert_eq!(pick_paste_tool(|t| t == "ydotool"), Some("ydotool")); }
#[test] fn none_when_missing() { assert_eq!(pick_paste_tool(|_| false), None); }
```
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement.** Always set the clipboard first via `arboard::Clipboard::set_text` (on Wayland prefer `wl-copy` via `std::process::Command` when `arboard` fails). Then sleep `paste_delay_ms` and synthesize paste:
  - Windows / X11: `enigo` Ctrl+V (`Key::Control` press, `Key::Unicode('v')` click, release).
  - Wayland: `wtype -M ctrl v -m ctrl` or `ydotool key 29:1 47:1 47:0 29:0`; if neither tool exists return `PasteOutcome::ClipboardOnly("install wtype or ydotool")`.
  - If synthesis fails, return `ClipboardOnly(err)` (text is already on the clipboard).
  `which`-style presence check: `Command::new(tool).arg("--version")` spawn succeeds.
- [ ] **Step 4:** Run unit tests; expect 3 passed.
- [ ] **Step 5: Manual check:** add hidden `paste-test "text"` subcommand (sleeps 3 s so you can focus a text box, then pastes). Verify on Windows and one Linux session.
- [ ] **Step 6:** Commit `feat(dictation): clipboard and paste output`.

### Task 3.3: Runtime wiring and `run`

**Files:** Create `src/runtime.rs`; modify `src/main.rs`, `src/lib.rs`.

**Consumes:** `Controller`, `Input`, `Action`, `Status` (3.1); `Recorder`/`CpalRecorder`, `Recording` (2.x); `Engine` (1.1); `Paster`/`system_paster` (3.2); `Config` (0.2).

**Produces:**
```rust
pub struct Runtime { pub tx: std::sync::mpsc::Sender<RuntimeMsg> }
pub enum RuntimeMsg { Input(Input), Reload, Shutdown }
pub fn spawn(config: Config, on_status: Box<dyn Fn(Status) + Send>, on_notify: Box<dyn Fn(String) + Send>) -> Runtime
```
Design: one coordinator thread owns `Controller` and `Receiver<RuntimeMsg>`. Executing actions:
- `StartCapture` → `recorder.start()`; on `Err` feed `Input::CaptureFailed`.
- `StopCapture` → `recorder.stop()`; feed `Input::Captured(rec)` or `CaptureFailed`.
- `Transcribe(rec)` → send to the engine thread over a channel; the engine thread replies by sending `RuntimeMsg::Input(Input::Transcribed(..))` on a clone of `tx`.
- `Paste(text)` → `paster.paste(text)`; `ClipboardOnly(reason)` → `on_notify`.
- `SetStatus` / `Notify` → callbacks.
Engine thread resolves the model dir with `resolve_model_dir(config.model_dir, exe_dir, dirs::data_dir())`; `None` → `EngineFailed("model not found; expected <paths>")` (no download is ever attempted), otherwise loads and sends `EngineReady`/`EngineFailed`. Recorder `max_secs` auto-stop: a timer thread started on `StartCapture` sends `Input::Release` after `max_record_secs` if still recording (controller ignores a stray release).

- [ ] **Step 1:** Implement `runtime.rs` as above.
- [ ] **Step 2:** In `main.rs`, `run` (default) subcommand: load config, `spawn`, print status/notify to stderr, and read stdin lines `s` → `Input::Press`, `e` → `Input::Release`, `q` → Shutdown (temporary dev trigger; removed in Sprint 5).
- [ ] **Step 3: Manual E2E:** `cargo run -p dictation-service -- run`, wait for the "Idle" status line, type `s` + Enter, speak, type `e` + Enter. Because the terminal has focus when you type, set `paste_delay_ms = 4000` in the config for this check: after typing `e`, click into a text editor before the delay ends. Expected: text pasted there and on the clipboard (Sprint 4 removes this awkwardness).
- [ ] **Step 4:** `cargo test -p dictation-service` all green; commit `feat(dictation): runtime wiring`.

---

# Sprint 4 — Control surface (IPC + CLI)

**Exit criteria:** with `dictation run` active, `dictation start`, `dictation stop`, `dictation toggle`, `dictation status`, `dictation reload` work; unit tests for protocol pass. This is also the Wayland/Hyprland path.

### Task 4.1: Control protocol and server/client

**Files:** Modify `Cargo.toml` (`interprocess = "2"`); create `src/control.rs`, `tests/control.rs`; modify `src/main.rs`, `src/runtime.rs`.

**Produces:**
```rust
pub enum Command { Start, Stop, Toggle, Status, Reload }
impl Command { pub fn parse(line: &str) -> Option<Command>; pub fn as_line(&self) -> &'static str }
pub fn socket_name() -> String            // Linux: $XDG_RUNTIME_DIR/dictation.sock; Windows: "dictation"
pub fn serve(handler: impl Fn(Command) -> String + Send + 'static) -> std::io::Result<()>   // blocking accept loop; spawn on a thread
pub fn send(cmd: Command) -> std::io::Result<String>
```
Protocol: one line request (`start\n`), one line response (`ok`, `recording`, `idle`, `error: ...`).

- [ ] **Step 1: Failing tests**
```rust
use dictation_service::control::Command;
#[test] fn parses_known_commands_case_insensitively() {
    assert_eq!(Command::parse("START\n"), Some(Command::Start));
    assert_eq!(Command::parse(" toggle "), Some(Command::Toggle));
}
#[test] fn rejects_unknown() { assert_eq!(Command::parse("rm -rf"), None); }
#[test] fn round_trips_names() {
    for c in [Command::Start, Command::Stop, Command::Toggle, Command::Status, Command::Reload] {
        assert_eq!(Command::parse(c.as_line()), Some(c));
    }
}
```
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement** using `interprocess::local_socket` (`GenericNamespaced` on Windows, `GenericFilePath` on Linux). Remove a stale Linux socket file at server start; refuse to start a second instance (bind failure → "already running" error). Set socket permissions to the user only (`0o600` on Linux).
- [ ] **Step 4:** Wire `Start`/`Stop` → `Input::Press`/`Release`; `Toggle` uses the controller's recording state (expose `Controller::is_recording()`; add a test for it); `Status` returns the current `Status` name (runtime keeps the last status in an `Arc<Mutex<Status>>`); `Reload` → `RuntimeMsg::Reload` which reloads config and, if the model changed, reloads the engine (set controller back to Loading via `Input::EngineReady` after load; add `Controller::set_loading()`).
- [ ] **Step 5:** `main.rs`: `start|stop|toggle|status|reload` subcommands call `control::send`, print the reply, exit non-zero if the service isn't running.
- [ ] **Step 6: Manual E2E:** run `dictation run` in one terminal; in another `dictation start`, speak, `dictation stop` → text pasted. Remove the stdin dev trigger from 3.3.
- [ ] **Step 7:** Commit `feat(dictation): control socket and CLI`.

### Task 4.2: Hyprland/Omarchy docs

**Files:** Create `docs/dictation.md` (initial section).

- [ ] Document: install requirements (`wtype`), the Hyprland bindings
```
bind  = CTRL, SPACE, exec, dictation start
bindr = CTRL, SPACE, exec, dictation stop
```
`exec-once = dictation run`, and the Wayland caveats. Commit `docs(dictation): Wayland setup`.

---

# Sprint 5 — In-process hotkeys (Windows + X11, optional evdev)

**Exit criteria:** holding Ctrl+Space on Windows and X11 records/transcribes/pastes; chord logic fully unit-tested.

### Task 5.1: Chord parsing and tracking (pure)

**Files:** Create `src/chord.rs`, `tests/chord.rs`; modify `src/lib.rs`.

**Produces:**
```rust
pub struct Chord { /* normalized key names */ }
impl Chord { pub fn parse(s: &str) -> Result<Chord, String>; }
pub enum HotkeyEvent { Pressed, Released }
pub struct ChordTracker { /* private */ }
impl ChordTracker { pub fn new(chord: Chord) -> Self;
                    pub fn feed(&mut self, key: &str, down: bool) -> Option<HotkeyEvent>; }
```
Key names are lowercase strings: `ctrl`, `shift`, `alt`, `meta`, `space`, `a`..`z`, `0`..`9`, `f1`..`f24`, etc. Left/right variants (`ControlLeft`/`ControlRight`) are normalized by the backend to `ctrl`. Rules: `Pressed` fires once when every chord key is down; `Released` fires once when any chord key goes up after a `Pressed`; key auto-repeat (down while already down) is ignored.

- [ ] **Step 1: Failing tests**
```rust
use dictation_service::chord::{Chord, ChordTracker, HotkeyEvent::*};

fn t() -> ChordTracker { ChordTracker::new(Chord::parse("ctrl+space").unwrap()) }

#[test] fn press_then_release() {
    let mut c = t();
    assert_eq!(c.feed("ctrl", true), None);
    assert_eq!(c.feed("space", true), Some(Pressed));
    assert_eq!(c.feed("space", false), Some(Released));
}
#[test] fn release_by_either_key() {
    let mut c = t();
    c.feed("ctrl", true); c.feed("space", true);
    assert_eq!(c.feed("ctrl", false), Some(Released));
    assert_eq!(c.feed("space", false), None);
}
#[test] fn autorepeat_does_not_retrigger() {
    let mut c = t();
    c.feed("ctrl", true);
    assert_eq!(c.feed("space", true), Some(Pressed));
    assert_eq!(c.feed("space", true), None);
}
#[test] fn space_before_ctrl_also_works() {
    let mut c = t();
    assert_eq!(c.feed("space", true), None);
    assert_eq!(c.feed("ctrl", true), Some(Pressed));
}
#[test] fn unrelated_keys_are_ignored() {
    let mut c = t();
    assert_eq!(c.feed("a", true), None);
}
#[test] fn parse_rejects_empty_and_normalizes_case() {
    assert!(Chord::parse("").is_err());
    assert!(Chord::parse("Ctrl + SPACE").is_ok());
}
```
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement** with a `BTreeSet<String>` for the chord and `held` set plus `active: bool`.
- [ ] **Step 4:** Run, expect 6 passed.
- [ ] **Step 5:** Commit `feat(dictation): chord tracker`.

### Task 5.2: rdev backend and backend selection

**Files:** Modify `Cargo.toml` (`rdev = "0.5"`); create `src/hotkey/mod.rs`, `src/hotkey/rdev_backend.rs`; modify `src/runtime.rs`, `src/main.rs`.

**Produces:**
```rust
pub trait HotkeyBackend: Send {
    fn name(&self) -> &'static str;
    fn spawn(self: Box<Self>, chord: Chord, on_event: Box<dyn Fn(HotkeyEvent) + Send>) -> Result<(), String>;
}
pub fn select(backend_pref: Option<&str>) -> Option<Box<dyn HotkeyBackend>>
```
`select`: Windows → rdev; Linux X11 (`!is_wayland()`) → rdev; Linux Wayland → `evdev` if feature enabled and `/dev/input` readable, else `None` (control CLI is the path; log an info message pointing to `docs/dictation.md`).

- [ ] **Step 1:** Implement `rdev_backend.rs`: `rdev::listen` on its own thread; map `Key::ControlLeft|ControlRight → "ctrl"`, `ShiftLeft|ShiftRight → "shift"`, `Alt|AltGr → "alt"`, `MetaLeft|MetaRight → "meta"`, `Space → "space"`, letters/digits/F-keys by name; feed `ChordTracker`; forward events as `Input::Press`/`Release`.
- [ ] **Step 2:** Wire in `runtime::spawn`; log which backend is active.
- [ ] **Step 3: Manual E2E on Windows:** hold Ctrl+Space in Notepad, speak, release → text appears. Repeat in a browser text box. Check behaviour in an elevated window (expected: no paste, documented).
- [ ] **Step 4: Manual E2E on X11.**
- [ ] **Step 5:** Commit `feat(dictation): rdev hotkey backend`.

### Task 5.3: Optional evdev backend (Linux, feature `evdev`)

**Files:** Create `src/hotkey/evdev_backend.rs` (`#[cfg(all(target_os = "linux", feature = "evdev"))]`).

- [ ] Enumerate `/dev/input/event*`, select devices with `KEY_SPACE` and `KEY_LEFTCTRL` capability, read events on one thread per device, map codes (`KEY_LEFTCTRL|KEY_RIGHTCTRL → ctrl`, etc.) and feed the same `ChordTracker`. Ignore `value == 2` (autorepeat). On permission error return `Err("add your user to the input group")`.
- [ ] Manual check on a Wayland session with the user in the `input` group. Document the privacy tradeoff in `docs/dictation.md`.
- [ ] Commit `feat(dictation): evdev hotkey backend`.

---

# Sprint 6 — Tray and settings

**Exit criteria:** tray icon shows loading/idle/recording/transcribing/error; menu works; `dictation settings` edits and saves config and the running service picks it up.

### Task 6.1: Tray icon and notifications

**Files:** Modify `Cargo.toml` (`tray-icon = "0.19"`, `tao = "0.30"`, `notify-rust = "4"`); create `src/tray.rs`, `assets/*.png` (five 32×32 icons: loading, idle, recording, transcribing, error); modify `src/main.rs`.

**Consumes:** `Runtime`, `RuntimeMsg`, `Status` (3.3 / 3.1).

- [ ] **Step 1:** `main.rs` `run`: build the runtime, then run a `tao` event loop on the main thread owning the `TrayIcon`. `on_status` posts a user event to swap the icon and tooltip; `on_notify` calls `notify-rust` (fall back to `log::warn!` when no notification daemon).
- [ ] **Step 2:** Menu: `Status: <state>` (disabled), `Pause/Resume` (ignores hotkey events while paused: `Runtime` gets `RuntimeMsg::SetPaused(bool)` and the coordinator drops `Press` inputs), `Settings…` (spawns `std::env::current_exe() settings`), `Quit` (sends `Shutdown`, exits the loop).
- [ ] **Step 3:** On Linux the tray needs GTK + libappindicator; document in `docs/dictation.md`. If tray creation fails, log a warning and continue headless (the service must still work under a compositor without a tray).
- [ ] **Step 4: Manual check:** the icon changes through the full cycle; Quit exits cleanly; Pause blocks recording.
- [ ] **Step 5:** Commit `feat(dictation): tray icon, status and notifications`.

### Task 6.2: Settings window

**Files:** Modify `Cargo.toml` (`eframe = "0.29"`); create `src/settings.rs`; modify `src/main.rs`.

**Produces:** subcommand `dictation settings` (separate process).

- [ ] **Step 1:** UI fields: hotkey (button "Record new hotkey" capturing the next chord via egui key events, producing a `Chord`-parsable string such as `ctrl+space`; shown only when not on Wayland, otherwise show the compositor binding help text), model (ComboBox listing the bundled `ivrit-turbo` plus any other model folders found under `models/`, and a "Model folder" path field → `model_dir`), language (ComboBox with `auto` plus a short list of common ISO codes and a free text field validated by `Config::validate`), input device (ComboBox from `cpal` device names + "Default"), Save / Cancel.
- [ ] **Step 2:** Save: `Config::validate()` → on error show it inline and don't write; on success `save_to(default_path())`, then call `control::send(Command::Reload)` (ignore "not running").
- [ ] **Step 3:** Add a pure helper `fn validate_form(&FormState) -> Result<Config, String>` with unit tests for the language/model rules (reusing `Config::validate`).
- [ ] **Step 4: Manual check:** change the model folder to a second model (or break the path), Save; tray shows Loading then Idle (or Error with the expected-path message) and the next dictation uses the new model.
- [ ] **Step 5:** Commit `feat(dictation): settings window`.

---

# Sprint 7 — Hardening, docs, release

**Exit criteria:** manual QA checklist passes on Windows and one Linux session (X11 or Hyprland); README and `docs/dictation.md` complete; workspace tests green.

### Task 7.1: Error handling pass

**Files:** Modify `src/runtime.rs`, `src/audio.rs`, `src/output.rs`.

- [ ] **Step 1:** Mic lost mid-recording: `stop()` returns `Err` → `CaptureFailed` → notification; next press re-opens the device (recorder rebuilds its stream on every `start`).
- [ ] **Step 2:** Model missing or unloadable at startup: `EngineFailed` → Error status + notification naming the searched paths (no download is attempted); add a tray item "Retry model load" that sends `Reload`.
- [ ] **Step 3:** Single-instance: starting `dictation run` twice prints "already running" and exits 1 (from the control bind failure).
- [ ] **Step 4:** Add a controller test that `EngineFailed` followed by `Press` does not start capture (status stays Error) and fix if needed.
- [ ] **Step 5:** Commit `fix(dictation): error handling pass`.

### Task 7.2: Docs and manual QA

**Files:** Create/finish `docs/dictation.md`; modify `README.md` (Layout + Building).

- [ ] **Step 1:** Finish `docs/dictation.md`: install/build, config file reference (every field, default, valid values), hotkey setup per platform (Windows, X11, Wayland/Hyprland incl. Omarchy), `wtype`/`ydotool`, evdev option and privacy note, known limitations (elevated windows, terminals need Ctrl+Shift+V, Wayland tray requirements, Ctrl+Space collisions with IMEs), troubleshooting.
- [ ] **Step 2:** README: add the crate under **Layout** and the build command `cargo build --release -p dictation-service`.
- [ ] **Step 3: Manual QA checklist** (record results in the PR):
  - [ ] Hold/release records and pastes in Notepad / browser / VS Code (Windows)
  - [ ] Same on X11 and on Hyprland via binds
  - [ ] Clipboard contains the text afterwards
  - [ ] Press during transcription queues and pastes in order; 4th rapid recording drops with a notice
  - [ ] Taps < 300 ms do nothing; silence pastes nothing
  - [ ] Unplug mic mid-recording → notification, next press works
  - [ ] Changing language to `he` in settings and dictating Hebrew
  - [ ] Peak VRAM and idle RAM with the bundled model noted in docs (must stay within 4 GB VRAM)
  - [ ] Airgap proof: with networking disabled (Windows: disable adapters; Linux: `unshare -n` or unplug), cold start, dictate Hebrew, everything works; and a network capture/firewall log shows zero outbound connections from `dictation`
- [ ] **Step 4:** Run `cargo test --workspace` and `cargo clippy -p dictation-service -- -D warnings`; fix findings.
- [ ] **Step 5:** Commit `docs(dictation): usage guide and README`.

---

### Task 7.3: Offline installer with the bundled model

**Files:** Create `packaging/` (Windows installer script, Linux packaging notes), `docs/dictation.md` (install section); modify `README.md`.

- [ ] **Step 1:** Windows installer (NSIS, same tool family as the existing app) installs `dictation.exe` plus `models\ivrit-turbo\{config.json,model.safetensors}` next to it (or under `%ProgramData%\dictation\models` for per-machine installs, which `resolve_model_dir` also checks via the data dir). Installer size is ~3.2 GB (f32) or smaller if the Task 1.3 decision ships f16 weights; note the NSIS 2 GB limit — use a separate model payload (second archive, or an MSI/zip bundle) if the file exceeds it.
- [ ] **Step 2:** Linux: a tarball (and optional deb/rpm) with `bin/dictation` and `share/dictation/models/ivrit-turbo/`; the service also checks `/usr/share/dictation/models/ivrit-turbo` — add that path to `resolve_model_dir` with a test.
- [ ] **Step 3:** Ship a `SHA256SUMS` for the model files; add `dictation verify-model` (hidden) that recomputes and compares the checksum so IT can validate a transfer into the airgapped network. Unit-test the checksum helper on a small temp file.
- [ ] **Step 4:** Document how to carry the installer into an airgapped network and the model license (Apache-2.0, ivrit-ai attribution, from the model card) in `docs/dictation.md` and a `THIRD_PARTY_NOTICES` entry.
- [ ] **Step 5:** Test on a clean VM with networking disabled: install, start, dictate Hebrew.
- [ ] **Step 6:** Commit `feat(dictation): offline installer with bundled model`.

---

## Sprint summary

| Sprint | Delivers | Demo |
|---|---|---|
| 0 | Crate, spec fixes, config | `dictation --help`, config tests |
| 1 | Offline engine + **4 GB VRAM spike (decision gate)** | `transcribe-file` prints Hebrew text; VRAM table |
| 2 | Mic capture | `record-test` shows level |
| 3 | Controller, output, runtime | stdin-triggered dictation pastes text |
| 4 | Control socket + CLI | Hyprland binds work end to end |
| 5 | Chord + rdev (+ evdev) | Hold Ctrl+Space on Windows/X11 |
| 6 | Tray + settings | Full UI, live reload |
| 7 | Hardening, docs, offline installer with bundled model | QA checklist incl. airgap proof |

Sprints 0–4 already give a complete working service on Wayland/Hyprland (via binds); Sprint 5 adds the native hotkey for Windows/X11.
