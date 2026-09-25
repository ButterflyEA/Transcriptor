# Whisper-Burn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch, pure-Rust Whisper transcription library (`whisper-burn`) + CLI (`whisper-burn-cli`) on Burn that loads official `openai/whisper-*` safetensors checkpoints and reproduces the reference transcription for the deterministic (temperature=0) path.

**Architecture:** Cargo workspace: a lib crate (PCM-in, structured-text-out, backend-generic) + a thin bin crate (file decode + download). Model/tokenizer/mel/safetensors/decoding are hand-implemented for fidelity; boring IO uses crates (symphonia, rubato, ureq, clap). All model+decoding code is generic over `B: Backend`; the concrete backend (wgpu default, ndarray fallback) is chosen via Cargo features.

**Tech Stack:** Rust (edition 2024), burn 0.16 + burn-wgpu + burn-ndarray, thiserror, serde/serde_json, regex, base64, ureq, symphonia, rubato, clap.

**Spec:** `docs/superpowers/specs/2026-09-19-whisper-burn-design.md`

## Global Constraints

- **Dims from config.json, never hardcoded per size**; `ModelSize` only maps to repo id + validates dims against the expected family.
- **Safetensors F32-only reader**; the weight map **must be fully drained** by model construction (extra/unconsumed weights are `UnexpectedWeight` errors).
- **Fidelity constants:** conv1 `k3 pad1`, conv2 `k3 s2 pad1` → 1500 encoder tokens; GELU = erf; LayerNorm eps `1e-5`; `q·kᵀ/√d_head` softmax; log-mel `clamp(1e-10)`, `log10`, `max(x, max−8)`, `(x+4)/4`; special ids `eot=50256`, `sot=50257`, `translate=50358`, `transcribe=50359`, `startoflm=50360`, `startofprev=50361`, `nospeech=50362`, `notimestamps=50363`, `timestamp_begin=50364`; language token = `50258 + sorted-index`; `yue` append-only as `n_vocab-1` when `n_vocab==51866`.
- **v1 scope:** greedy (t=0) + beam, language detection, no-speech, timestamps, 30 s chunking, `condition_on_previous_text`. **No** stochastic sampling (v2), no torch `.bin`.
- **Public API** exposes `WgpuWhisper` / `CpuWhisper` aliases so callers never see `B`.
- **Tests:** numeric/format modules are TDD against golden values; checkpoint-downloading tests are `#[ignore]`.
- **Commits:** one per task, conventional style (`feat:`/`test:`/`docs:`).
- **Burn 0.16 API conventions used throughout:** shapes are arrays (`Tensor::ones([s], &dev)`, `Shape::new([...])`, no `-1` inference in `reshape` — use explicit dims from `shape().dims()`); elementwise custom math uses `Tensor::map` on `FloatElem`; broadcasting follows Burn's rules (`.unsqueeze()`). Where the exact signature differs on the installed Burn version, keep the *math* pinned by the task's golden test and adapt the call, noting the change in the commit.

## File Structure (final)

```
Cargo.toml                          # [workspace] only
crates/whisper-burn/
  Cargo.toml
  src/lib.rs            # crate root + re-exports
  src/error.rs          # Error enum + Result
  src/config.rs         # ModelDimensions, ModelSize
  src/backends.rs       # CpuWhisper/WgpuWhisper aliases + device helpers
  src/download.rs       # ureq download + cache dir (feature "weights")
  src/weights/mod.rs    # module (safetensors + loader)
  src/weights/safetensors.rs   # safetensors file reader
  src/weights/loader.rs        # WeightMap + take_1d/2d/3d + finish drain
  src/tokenizer/mod.rs  # tiktoken + whisper
  src/tokenizer/tiktoken.rs    # byte-level BPE core
  src/tokenizer/whisper.rs     # whisper special tokens/languages
  src/features/mod.rs   # stft + mel + window
  src/features/stft.rs  # DFT STFT (torch center=True semantics)
  src/features/mel.rs   # baked filterbank + log-mel
  src/features/window.rs # 30s chunking
  src/audio/mod.rs      # decode + resample (feature "audio")
  src/audio/decode.rs   # symphonia -> f32 mono
  src/audio/resample.rs # rubato -> 16kHz
  src/model/mod.rs      # attention/block/encoder/decoder/whisper/ops
  src/model/ops.rs      # Linear + LayerNorm + GELU(erf) helpers
  src/model/attention.rs# MultiHeadAttention
  src/model/block.rs    # ResidualAttentionBlock
  src/model/encoder.rs
  src/model/decoder.rs
  src/model/whisper.rs  # Whisper + load/from_pretrained/transcribe
  src/transcribe.rs     # transcribe orchestration
  src/decoding/mod.rs   # greedy + beam + lang + segments
  src/decoding/greedy.rs# greedy temperature-0 loop + suppression
  src/decoding/beam.rs  # beam search
  src/decoding/lang.rs  # detect_language
  src/decoding/segments.rs # timestamp parsing + segment assembly
  tests/*.rs            # unit/integration tests per task
  tests/fixtures/       # tiny_config.json, mini.tiktoken, golden refs
  assets/mel_filters_80.bin, mel_filters_128.bin
crates/whisper-burn-cli/src/main.rs
scripts/extract_mel_filters.py      # one-time dev script (not built)
scripts/cement_reference.py         # one-time golden cementing (not built)
```

---

### Task 1: Workspace scaffold

**Files:**
- Create: `Cargo.toml` at root (replace existing package Cargo.toml); `crates/whisper-burn/Cargo.toml`; `crates/whisper-burn/src/lib.rs`; `crates/whisper-burn-cli/Cargo.toml`; `crates/whisper-burn-cli/src/main.rs`
- Replace/remove old `src/` package
- Add `.gitignore` with `/target`, `Cargo.lock` (keep lock committed tracked; add `/crates/*/target`)

**Interfaces:**
- Produces: workspace that builds via `cargo build --workspace` and runs `cargo test --workspace` green; `whisper_burn` lib crate; `whisper-burn-cli` binary.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/src/lib.rs` placeholder:
```rust
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
```

`crates/whisper-burn-cli/src/main.rs`:
```rust
fn main() {
    println!("whisper-burn-cli {}", whisper_burn::version());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo build --workspace`
Expected: FAIL — crates not found, old package conflicts.

- [ ] **Step 3: Write the implementation**

Root `Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/whisper-burn", "crates/whisper-burn-cli"]

[workspace.package]
version = "0.1.0"
edition = "2024"

[workspace.dependencies]
burn = "0.16"
burn-wgpu = { version = "0.16", default-features = false }
burn-ndarray = { version = "0.16", default-features = false, features = ["std"] }
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
regex = "1"
base64 = "0.22"
ureq = { version = "2" }
clap = { version = "4", features = ["derive"] }
symphonia = { version = "0.5", features = ["all"] }
rubato = "0.16"
```

`crates/whisper-burn/Cargo.toml`:
```toml
[package]
name = "whisper-burn"
version.workspace = true
edition.workspace = true

[features]
default = ["wgpu"]
wgpu = ["dep:burn-wgpu"]
ndarray = ["dep:burn-ndarray"]
audio = ["dep:symphonia", "dep:rubato"]
std = []

[dependencies]
burn.workspace = true
burn-wgpu = { workspace = true, optional = true }
burn-ndarray = { workspace = true, optional = true }
thiserror.workspace = true
serde.workspace = true
serde_json.workspace = true
regex.workspace = true
base64.workspace = true
ureq.workspace = true
symphonia = { workspace = true, optional = true }
rubato = { workspace = true, optional = true }
```

`crates/whisper-burn-cli/Cargo.toml`:
```toml
[package]
name = "whisper-burn-cli"
version.workspace = true
edition.workspace = true

[dependencies]
whisper-burn = { path = "../whisper-burn", features = ["audio"] }
clap.workspace = true
```

Remove old root `src/` and old root `Cargo.toml` package (git mv contents as needed).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo build --workspace; cargo test --workspace`
Expected: PASS; `cargo run -p whisper-burn-cli` prints version.

- [ ] **Step 5: Commit**

```bash
git add -u; git add Cargo.toml crates/ .gitignore
git commit -m "feat: scaffold workspace with lib and cli crates"
```

---

### Task 2: Error type and Result alias

**Files:**
- Create: `crates/whisper-burn/src/error.rs`
- Modify: `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces: `pub enum Error` (thiserror) + `pub type Result<T> = std::result::Result<T, Error>`. Variants: `Io { path, source }`, `Download(String)`, `MissingWeights { path }`, `UnsupportedFormat(String)`, `ConfigParse(String)`, `MissingWeight(String)`, `UnexpectedWeight(String)`, `ShapeMismatch { name, expected, got }`, `Decoder(String)`, `Tokenizer(String)`, `AudioDecode(String)`, `Resample(String)`.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/src/error.rs` will contain a unit-test module at the bottom. The module is added together with the implementation in Step 3; the test compiles only once the type exists:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_readable_errors() {
        let e = Error::ShapeMismatch {
            name: "enc.conv1.weight".into(),
            expected: vec![1, 2],
            got: vec![3, 4],
        };
        assert!(e.to_string().contains("enc.conv1.weight"));
        let e = Error::MissingWeight("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn io_error_converts() {
        let e: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(e, Error::Io { .. }));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --lib`
Expected: FAIL — `error` module not found.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/error.rs`:
```rust
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },
    #[error("download failed: {0}")]
    Download(String),
    #[error("missing weights file: {path}")]
    MissingWeights { path: PathBuf },
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("config parse error: {0}")]
    ConfigParse(String),
    #[error("missing weight in checkpoint: {0}")]
    MissingWeight(String),
    #[error("unexpected extra weight in checkpoint: {0}")]
    UnexpectedWeight(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch { name: String, expected: Vec<usize>, got: Vec<usize> },
    #[error("decoder error: {0}")]
    Decoder(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("audio decode error: {0}")]
    AudioDecode(String),
    #[error("resample error: {0}")]
    Resample(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```
Append the test module from Step 1 to this file.

`crates/whisper-burn/src/lib.rs`:
```rust
pub mod error;
pub use error::{Error, Result};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --lib`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: add Error enum and Result alias"
```

---

### Task 3: ModelDimensions, config.json parsing, ModelSize

**Files:**
- Create: `crates/whisper-burn/src/config.rs`
- Create: `crates/whisper-burn/tests/fixtures/tiny_config.json`
- Modify: `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces:
  - `#[derive(Clone, Debug, PartialEq)] pub struct ModelDimensions { pub n_mels, pub n_audio_layer, pub n_text_layer, pub n_audio_state, pub n_text_state, pub n_head, pub n_vocab, pub n_audio_ctx, pub n_text_ctx }` (all `usize`) with `pub fn from_config_json(s: &str) -> Result<Self>`
  - `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum ModelSize { Tiny, Base, Small, Medium, Large, LargeV2, LargeV3, LargeV3Turbo }` with `fn repo_id(self) -> &'static str`, `fn expected_dims(self) -> Dims`, `fn validate(self, &ModelDimensions) -> Result<()>`, `impl std::fmt::Display`
  - `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct Dims { n_mels, n_vocab, n_audio_layer, n_text_layer }`

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/fixtures/tiny_config.json` (verbatim HF `openai/whisper-tiny/config.json`):
```json
{
  "activation_function": "gelu",
  "architectures": ["WhisperForConditionalGeneration"],
  "d_model": 384,
  "decoder_attention_heads": 6,
  "decoder_ffn_dim": 1536,
  "decoder_layers": 4,
  "decoder_start_token_id": 50258,
  "dropout": 0.0,
  "encoder_attention_heads": 6,
  "encoder_ffn_dim": 1536,
  "encoder_layerdrop": 0,
  "encoder_layers": 4,
  "init_std": 0.02,
  "is_encoder_decoder": true,
  "max_source_positions": 1500,
  "max_target_positions": 448,
  "model_type": "whisper",
  "num_hidden_layers": 4,
  "num_mel_bins": 80,
  "pad_token_id": 50256,
  "scale_embedding": false,
  "torch_dtype": "float32",
  "use_cache": true,
  "vocab_size": 51865
}
```

`crates/whisper-burn/tests/config.rs`:
```rust
use whisper_burn::config::{Dims, ModelDimensions, ModelSize};

const TINY: &str = include_str!("fixtures/tiny_config.json");

#[test]
fn parses_tiny_config() {
    let d = ModelDimensions::from_config_json(TINY).unwrap();
    assert_eq!(d.n_mels, 80);
    assert_eq!(d.n_audio_layer, 4);
    assert_eq!(d.n_text_layer, 4);
    assert_eq!(d.n_audio_state, 384);
    assert_eq!(d.n_text_state, 384);
    assert_eq!(d.n_head, 6);
    assert_eq!(d.n_vocab, 51865);
    assert_eq!(d.n_audio_ctx, 1500);
    assert_eq!(d.n_text_ctx, 448);
}

#[test]
fn missing_key_errors() {
    let bad = TINY.replace("\"num_mel_bins\": 80", "\"zzz\": 80");
    assert!(ModelDimensions::from_config_json(&bad).is_err());
}

#[test]
fn repo_ids_and_expected_dims() {
    assert_eq!(ModelSize::Tiny.repo_id(), "openai/whisper-tiny");
    assert_eq!(ModelSize::LargeV3.repo_id(), "openai/whisper-large-v3");
    assert_eq!(ModelSize::LargeV3Turbo.repo_id(), "openai/whisper-large-v3-turbo");
    assert_eq!(
        ModelSize::LargeV3.expected_dims(),
        Dims { n_mels: 128, n_vocab: 51866, n_audio_layer: 32, n_text_layer: 32 }
    );
    assert_eq!(
        ModelSize::LargeV3Turbo.expected_dims(),
        Dims { n_mels: 128, n_vocab: 51866, n_audio_layer: 32, n_text_layer: 4 }
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test config`
Expected: FAIL — `config` module doesn't exist.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/config.rs`:
```rust
use crate::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_layer: usize,
    pub n_text_layer: usize,
    pub n_audio_state: usize,
    pub n_text_state: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub n_audio_ctx: usize,
    pub n_text_ctx: usize,
}

#[derive(Deserialize)]
struct HfConfig {
    #[serde(rename = "num_mel_bins")]
    n_mels: usize,
    #[serde(rename = "encoder_layers")]
    n_audio_layer: usize,
    #[serde(rename = "decoder_layers")]
    n_text_layer: usize,
    #[serde(rename = "d_model")]
    d_model: usize,
    #[serde(rename = "encoder_attention_heads")]
    n_head: usize,
    #[serde(rename = "vocab_size")]
    n_vocab: usize,
    #[serde(rename = "max_source_positions")]
    n_audio_ctx: usize,
    #[serde(rename = "max_target_positions")]
    n_text_ctx: usize,
}

impl ModelDimensions {
    pub fn from_config_json(s: &str) -> Result<Self> {
        let c: HfConfig = serde_json::from_str(s)
            .map_err(|e| Error::ConfigParse(format!("config.json: {e}")))?;
        Ok(Self {
            n_mels: c.n_mels,
            n_audio_layer: c.n_audio_layer,
            n_text_layer: c.n_text_layer,
            n_audio_state: c.d_model,
            n_text_state: c.d_model,
            n_head: c.n_head,
            n_vocab: c.n_vocab,
            n_audio_ctx: c.n_audio_ctx,
            n_text_ctx: c.n_text_ctx,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub n_mels: usize,
    pub n_vocab: usize,
    pub n_audio_layer: usize,
    pub n_text_layer: usize,
}

impl ModelSize {
    pub fn repo_id(self) -> &'static str {
        match self {
            ModelSize::Tiny => "openai/whisper-tiny",
            ModelSize::Base => "openai/whisper-base",
            ModelSize::Small => "openai/whisper-small",
            ModelSize::Medium => "openai/whisper-medium",
            ModelSize::Large => "openai/whisper-large",
            ModelSize::LargeV2 => "openai/whisper-large-v2",
            ModelSize::LargeV3 => "openai/whisper-large-v3",
            ModelSize::LargeV3Turbo => "openai/whisper-large-v3-turbo",
        }
    }

    pub fn expected_dims(self) -> Dims {
        match self {
            ModelSize::Tiny => Dims { n_mels: 80, n_vocab: 51865, n_audio_layer: 4, n_text_layer: 4 },
            ModelSize::Base => Dims { n_mels: 80, n_vocab: 51865, n_audio_layer: 6, n_text_layer: 6 },
            ModelSize::Small => Dims { n_mels: 80, n_vocab: 51865, n_audio_layer: 12, n_text_layer: 12 },
            ModelSize::Medium => Dims { n_mels: 80, n_vocab: 51865, n_audio_layer: 24, n_text_layer: 24 },
            ModelSize::Large | ModelSize::LargeV2 => Dims { n_mels: 80, n_vocab: 51865, n_audio_layer: 32, n_text_layer: 32 },
            ModelSize::LargeV3 => Dims { n_mels: 128, n_vocab: 51866, n_audio_layer: 32, n_text_layer: 32 },
            ModelSize::LargeV3Turbo => Dims { n_mels: 128, n_vocab: 51866, n_audio_layer: 32, n_text_layer: 4 },
        }
    }

    pub fn validate(&self, d: &ModelDimensions) -> Result<()> {
        let e = self.expected_dims();
        if d.n_mels != e.n_mels
            || d.n_vocab != e.n_vocab
            || d.n_audio_layer != e.n_audio_layer
            || d.n_text_layer != e.n_text_layer
        {
            return Err(Error::ConfigParse(format!(
                "{} config.json dims do not match expected {:?}",
                self.repo_id(),
                e
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelSize::Tiny => "Tiny",
            ModelSize::Base => "Base",
            ModelSize::Small => "Small",
            ModelSize::Medium => "Medium",
            ModelSize::Large => "Large",
            ModelSize::LargeV2 => "LargeV2",
            ModelSize::LargeV3 => "LargeV3",
            ModelSize::LargeV3Turbo => "LargeV3Turbo",
        })
    }
}
```

`crates/whisper-burn/src/lib.rs` — add:
```rust
pub mod config;
pub use config::ModelSize;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test config`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: parse ModelDimensions from config.json; add ModelSize"
```

---

### Task 4: Safetensors reader

**Files:**
- Create: `crates/whisper-burn/src/weights/mod.rs`
- Create: `crates/whisper-burn/src/weights/safetensors.rs`
- Modify: `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub fn read_safetensors(bytes: &[u8]) -> Result<SafeTensors>`
  - `pub struct SafeTensors { pub metadata: std::collections::HashMap<String, String>, pub tensors: Vec<TensorMeta> }`
  - `pub struct TensorMeta { pub name: String, pub dtype: DType, pub shape: Vec<usize>, pub data_offsets: (usize, usize) }`
  - `pub enum DType { F32, F16, BF16 }` with `Display`
  - `impl SafeTensors { pub fn by_name(&self, name: &str) -> Option<&TensorMeta>; pub fn all_tensor_names(&self) -> impl Iterator<Item = &str> }`
  - `pub fn slice_f32<'a>(meta: &TensorMeta, bytes: &'a [u8]) -> Result<&'a [f32]>`

Format: 8-byte LE `u64` header length `n`, then JSON header `{"__metadata__": {}, "<name>": {"dtype": "...", "shape": [...], "data_offsets": [a, b]}}`, then raw tensor bytes. `data_offsets` are relative to the start of the tensor-data section (right after the header).

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/safetensors.rs`:
```rust
use whisper_burn::weights::safetensors::{self, DType};

fn build(json: &str, payload: &[u8]) -> Vec<u8> {
    let n = json.len() as u64;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(payload);
    out
}

#[test]
fn parses_single_f32_tensor() {
    let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
    let payload = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("w").unwrap();
    assert_eq!(t.dtype.to_string(), "F32");
    assert_eq!(t.shape, vec![2, 3]);
    let f = safetensors::slice_f32(t, &file).unwrap();
    assert_eq!(f, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn rejects_non_f32() {
    let json = r#"{"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}"#;
    let payload = [0u8; 4];
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("w").unwrap();
    assert!(matches!(t.dtype, DType::F16));
    assert!(safetensors::slice_f32(t, &file).is_err());
}

#[test]
fn rejects_malformed() {
    assert!(safetensors::read_safetensors(&[0u8; 8]).is_err());
    assert!(safetensors::read_safetensors(b"nope").is_err());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test safetensors`
Expected: FAIL — `weights` module not found.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/weights/mod.rs`:
```rust
pub mod safetensors;
```

`crates/whisper-burn/src/weights/safetensors.rs`:
```rust
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        })
    }
}

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct SafeTensors {
    pub metadata: HashMap<String, String>,
    pub tensors: Vec<TensorMeta>,
}

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    data_offsets: Option<[usize; 2]>,
}

impl SafeTensors {
    pub fn by_name(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn all_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }
}

pub fn read_safetensors(bytes: &[u8]) -> Result<SafeTensors> {
    if bytes.len() < 8 {
        return Err(Error::Download("safetensors: too short".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let json_start = 8;
    let json_end = json_start
        .checked_add(n)
        .ok_or_else(|| Error::Download("safetensors: header overflow".into()))?;
    if json_end > bytes.len() {
        return Err(Error::Download("safetensors: truncated header".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[json_start..json_end])
        .map_err(|e| Error::Download(format!("safetensors header: {e}")))?;
    let obj = header.as_object().ok_or_else(|| Error::Download("safetensors: header not object".into()))?;

    let mut metadata = HashMap::new();
    if let Some(m) = obj.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                metadata.insert(k.clone(), s.to_string());
            }
        }
    }

    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let e: HeaderEntry = serde_json::from_value(v.clone())
            .map_err(|e| Error::Download(format!("safetensors tensor {name}: {e}")))?;
        let dtype = match e.dtype.as_deref() {
            Some("F32") => DType::F32,
            Some("F16") => DType::F16,
            Some("BF16") => DType::BF16,
            other => return Err(Error::UnsupportedFormat(format!(
                "safetensors dtype {other:?} for {name}"
            ))),
        };
        let shape = e.shape.ok_or_else(|| Error::Download(format!("safetensors {name}: no shape")))?;
        let off = e.data_offsets.ok_or_else(|| Error::Download(format!("safetensors {name}: no offsets")))?;
        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            data_offsets: (off[0], off[1]),
        });
    }

    Ok(SafeTensors { metadata, tensors })
}

pub fn slice_f32<'a>(meta: &TensorMeta, bytes: &'a [u8]) -> Result<&'a [f32]> {
    if meta.dtype != DType::F32 {
        return Err(Error::UnsupportedFormat(format!(
            "tensor {} is {} (F32 required)",
            meta.name, meta.dtype
        )));
    }
    if bytes.len() < 8 {
        return Err(Error::Download("safetensors: too short".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8 + n;
    let (a, b) = meta.data_offsets;
    let start = data_start.checked_add(a).ok_or_else(|| Error::Download("offset overflow".into()))?;
    let end = data_start.checked_add(b).ok_or_else(|| Error::Download("offset overflow".into()))?;
    if end > bytes.len() {
        return Err(Error::Download("tensor out of bounds".into()));
    }
    let n_bytes = end - start;
    if n_bytes % 4 != 0 {
        return Err(Error::UnsupportedFormat(format!(
            "tensor {} byte length {} not multiple of 4",
            meta.name, n_bytes
        )));
    }
    // SAFETY: range bounds-checked above; Vec<u8> allocations are 8-aligned so the
    // data slice starts at a 4-aligned offset, making the f32 read aligned.
    let ptr = bytes[start..].as_ptr() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n_bytes / 4) })
}
```

Modify `crates/whisper-burn/src/lib.rs`: add `pub mod weights;`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test safetensors`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: add F32 safetensors reader with strict dtype/offset validation"
```

---

### Task 5: Tokenizer core (tiktoken byte-level BPE)

**Files:**
- Create: `crates/whisper-burn/src/tokenizer/mod.rs`
- Create: `crates/whisper-burn/src/tokenizer/tiktoken.rs`
- Create: `crates/whisper-burn/tests/fixtures/mini.tiktoken`
- Modify: `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub struct CoreBpe { encoder, decoder, special_encoder, special_decoder }`
  - `impl CoreBpe { from_tiktoken(&[u8]), from_tiktoken_with_special(&[u8], &[(String,u32)]), n_vocab() -> u32, encode_ordinary(&str) -> Result<Vec<u32>>, encode_piece(&str) -> Result<Vec<u32>>, decode(&[u32]) -> Result<String>, special(&str) -> Option<u32> }`
  - `.tiktoken` file format: UTF-8 text lines `<base64 bytes> <rank>`; rank == token id.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/fixtures/mini.tiktoken`:
```
aA== 0
bA== 1
aGk= 2
aGVsbG8= 3
aGVsbG8h 4
IA== 5
aGVs 6
bG93 7
```
(decode: `aA` -> "j", `bA` -> "k", `aGk=` -> "hi", `aGVsbG8=` -> "hello", `aGVsbG8h` -> "hello!", `IA==` -> " ", `aGVs` -> "hel", `bG93` -> "low")

`crates/whisper-burn/tests/tiktoken.rs`:
```rust
use whisper_burn::tokenizer::tiktoken::CoreBpe;

const MINI: &str = include_str!("fixtures/mini.tiktoken");

#[test]
fn parses_and_decodes() {
    let bpe = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    assert_eq!(bpe.decode(&[3]).unwrap(), "hello");
    assert_eq!(bpe.decode(&[4]).unwrap(), "hello!");
    assert_eq!(bpe.n_vocab(), 8);
}

#[test]
fn encode_uses_bpe_merges() {
    let bpe = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    assert_eq!(bpe.encode_ordinary("hel").unwrap(), vec![6]);
    assert_eq!(bpe.encode_ordinary("low").unwrap(), vec![7]);
    // "hello" has no split; merges "hel"+"lo"? not present -> falls to byte rank 1 then whole
    let ids = bpe.encode_ordinary("hello").unwrap();
    assert!(ids.len() <= 2);
    assert_eq!(bpe.decode(&ids).unwrap(), "hello");
}

#[test]
fn special_tokens_are_numbered() {
    let bpe = CoreBpe::from_tiktoken_with_special(
        MINI.as_bytes(),
        &[("<|sot|>".into(), 8), ("<|eot|>".into(), 9)],
    ).unwrap();
    assert_eq!(bpe.special("<|sot|>"), Some(8));
    assert!(bpe.special("<|nope|>").is_none());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test tiktoken`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/tokenizer/mod.rs`:
```rust
pub mod tiktoken;
```

`crates/whisper-burn/src/tokenizer/tiktoken.rs`:
```rust
use crate::{Error, Result};
use base64::Engine as _;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct CoreBpe {
    pub encoder: HashMap<Vec<u8>, u32>,
    pub decoder: Vec<Vec<u8>>,
    pub special_encoder: HashMap<Vec<u8>, u32>,
    pub special_tokens: Vec<(String, u32)>,
}

fn pat_str_re() -> regex::Regex {
    regex::Regex::new(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )
    .expect("static regex")
}

impl CoreBpe {
    pub fn from_tiktoken(bytes: &[u8]) -> Result<Self> {
        Self::from_tiktoken_with_special(bytes, &[])
    }

    pub fn from_tiktoken_with_special(bytes: &[u8], specials: &[(String, u32)]) -> Result<Self> {
        use base64::engine::general_purpose::STANDARD as B64;
        let text = std::str::from_utf8(bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut encoder = HashMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (tok, rank) = line
                .rsplit_once(' ')
                .ok_or_else(|| Error::Tokenizer(format!("bad tiktoken line: {line}")))?;
            let rank: u32 = rank
                .trim()
                .parse()
                .map_err(|_| Error::Tokenizer(format!("bad rank in line: {line}")))?;
            let tok = B64.decode(tok).map_err(|e| Error::Tokenizer(format!("base64: {e}")))?;
            encoder.insert(tok, rank);
        }
        let mut decoder = vec![Vec::new(); encoder.len()];
        for (k, v) in &encoder {
            decoder[*v as usize] = k.clone();
        }
        let mut special_encoder = HashMap::new();
        for (tok, id) in specials {
            special_encoder.insert(tok.as_bytes().to_vec(), *id);
        }
        Ok(Self {
            encoder,
            decoder,
            special_encoder,
            special_tokens: specials.to_vec(),
        })
    }

    pub fn n_vocab(&self) -> u32 {
        (self.encoder.len() + self.special_encoder.len()) as u32
    }

    pub fn special(&self, token: &str) -> Option<u32> {
        self.special_encoder.get(token.as_bytes()).copied()
    }

    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let re = pat_str_re();
        for m in re.find_iter(text) {
            let p = m.as_str();
            if let Some(id) = self.special_encoder.get(p.as_bytes()) {
                out.push(*id);
            } else {
                out.extend(self.encode_piece(p)?);
            }
        }
        Ok(out)
    }

    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>> {
        let bytes = piece.as_bytes();
        if bytes.is_empty() {
            return Ok(vec![]);
        }
        if let Some(&id) = self.encoder.get(bytes) {
            return Ok(vec![id]);
        }
        Ok(byte_pair_merge(bytes, &self.encoder))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &t in tokens {
            match self.decoder.get(t as usize) {
                Some(b) => bytes.extend_from_slice(b),
                None => {
                    if let Some((tk, _)) = self.special_tokens.iter().find(|(_, id)| *id == t) {
                        bytes.extend_from_slice(tk.as_bytes());
                    } else {
                        return Err(Error::Tokenizer(format!("decode: unknown token id {t}")));
                    }
                }
            }
        }
        String::from_utf8(bytes).map_err(|e| Error::Tokenizer(format!("utf8: {e}")))
    }
}

fn byte_pair_merge(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.len() == 1 {
        return match ranks.get(piece) {
            Some(&r) => vec![r],
            None => vec![0],
        };
    }
    let rank_of_parts = |a: (usize, usize), b: (usize, usize)| -> Option<u32> {
        let mut key = Vec::with_capacity(a.1 + b.1);
        key.extend_from_slice(&piece[a.0..a.0 + a.1]);
        key.extend_from_slice(&piece[b.0..b.0 + b.1]);
        ranks.get(&key).copied()
    };
    let mut starts: Vec<usize> = (0..piece.len()).collect();
    let mut lens: Vec<usize> = vec![1; piece.len()];
    let mut parts_rank: Vec<Option<u32>> = (0..starts.len().saturating_sub(1))
        .map(|i| rank_of_parts((starts[i], lens[i]), (starts[i + 1], lens[i + 1])))
        .collect();

    while !parts_rank.is_empty() {
        let best = parts_rank
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .min_by_key(|(_, r)| *r);
        let i = match best {
            Some((i, _)) => i,
            None => break,
        };
        lens[i] += lens[i + 1];
        starts.remove(i + 1);
        lens.remove(i + 1);
        parts_rank.remove(i);
        if i < parts_rank.len() {
            parts_rank[i] = rank_of_parts((starts[i], lens[i]), (starts[i + 1], lens[i + 1]));
        }
        if i > 0 {
            parts_rank[i - 1] = rank_of_parts((starts[i - 1], lens[i - 1]), (starts[i], lens[i]));
        }
    }

    starts
        .iter()
        .zip(lens.iter())
        .map(|(&s, &l)| {
            let key = piece[s..s + l].to_vec();
            ranks.get(&key).copied().unwrap_or_else(|| {
                ranks.get(&key[..1]).copied().unwrap_or(0)
            })
        })
        .collect()
}
```

Note: whisper special-token strings (`<|sot|>`, `<|en|>`, timestamps) are passed via `from_tiktoken_with_special`; single-byte pieces resolve to their own token id, which exists in the full vocab.

Modify `crates/whisper-burn/src/lib.rs`: add `pub mod tokenizer;`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test tiktoken`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: add byte-level tiktoken BPE core with special tokens"
```

---

### Task 6: Whisper tokenizer (special tokens, languages, timestamps)

**Files:**
- Create: `crates/whisper-burn/src/tokenizer/whisper.rs`
- Modify: `crates/whisper-burn/src/tokenizer/mod.rs`

**Interfaces:**
- Produces:
  - `pub struct TextTokenizer { pub core: CoreBpe, pub n_vocab: u32, pub n_audio_ctx: usize, pub timestamp_begin: u32, pub languages: Vec<&'static str>, pub language_ids: HashMap<&'static str, u32> }`
  - `impl TextTokenizer { pub fn new(bytes: &[u8], n_vocab: u32, n_audio_ctx: usize) -> Result<Self>, pub fn sot(&self), eot(&self), translate(&self), transcribe(&self), startoflm(&self), startofprev(&self), nospeech(&self), notimestamps(&self), same_language_flag? none, pub fn timestamp_token(&self, time: f64) -> u32, pub fn timestamp_tokens(&self, start: f64, end: f64) -> Vec<u32>, pub fn language_token(&self, code: &str) -> Option<u32>, pub fn encode(&self, text: &str) -> Result<Vec<u32>>, pub fn decode(&self, tokens: &[u32]) -> Result<String>, pub fn valid_language_tokens(&self) -> Vec<u32>, pub fn sot_sequence(&self, lang: Option<&str>, task: Task, timestamps: bool) -> Vec<u32> }`
  - `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Task { Transcribe, Translate }`
  - Special-token table built from ws constants + language tokens + timestamp range.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/src/tokenizer/whisper.rs` unit tests (run with `cargo test -p whisper-burn --lib`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tiktoken::CoreBpe;

    fn tok(n_vocab: u32) -> TextTokenizer {
        // Use the real multilingual vocab from the HF repo fixture if present;
        // otherwise build a tokenizer that only knows special tokens and a few words.
        let core = CoreBpe::from_tiktoken(include_bytes!("../..//tests/fixtures/mini.tiktoken")).unwrap();
        TextTokenizer::new(core_bytes(), n_vocab, 1500).unwrap()
    }
    // helper omitted: see lib test
}
```

Because building a realistic tokenizer needs the real vocabulary, put the test in `crates/whisper-burn/tests/whisper_tokenizer.rs` and construct the vocab lazily:

```rust
use whisper_burn::tokenizer::whisper::{Task, TextTokenizer};
use whisper_burn::tokenizer::tiktoken::CoreBpe;

const MINI: &str = include_str!("fixtures/mini.tiktoken");

#[test]
fn special_ids_are_fixed() {
    let core = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    // mini vocab has 8 tokens; append whisper specials
    let mut specials = vec![
        ("<|endoftext|>".to_string(), 50256),
        ("<|startoftranscript|>".to_string(), 50257),
        ("<|translate|>".to_string(), 50358),
        ("<|transcribe|>".to_string(), 50359),
        ("<|startoflm|>".to_string(), 50360),
        ("<|startofprev|>".to_string(), 50361),
        ("<|nospeech|>".to_string(), 50362),
        ("<|notimestamps|>".to_string(), 50363),
    ];
    // language tokens en..yue
    let mut id = 50258u32;
    for code in TextTokenizer::LANGUAGES {
        let name = format!("<|{code}|>");
        specials.push((name, id));
        id += 1;
    }
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    let t = TextTokenizer::new_from_core(core, 51865, 1500).unwrap();

    assert_eq!(t.sot(), 50257);
    assert_eq!(t.eot(), 50256);
    assert_eq!(t.translate(), 50358);
    assert_eq!(t.transcribe(), 50359);
    assert_eq!(t.nospeech(), 50362);
    assert_eq!(t.notimestamps(), 50363);
    assert_eq!(t.timestamp_begin, 50364);
    assert_eq!(t.language_token("en"), Some(50258));
    assert_eq!(t.language_token("zh"), Some(50259));
    assert_eq!(t.timestamp_token(0.0), 50364);
    assert_eq!(t.timestamp_token(30.0), 51864);
}

#[test]
fn large_v3_yue_appends() {
    let core = CoreBpe::from_tiktoken(MINI.as_bytes()).unwrap();
    let mut specials = vec![
        ("<|startoftranscript|>".to_string(), 50257),
        ("<|en|>".to_string(), 50258),
        ("<|translate|>".to_string(), 50358),
        ("<|transcribe|>".to_string(), 50359),
        ("<|notimestamps|>".to_string(), 50363),
        ("<|yue|>".to_string(), 51865),
    ];
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    let t = TextTokenizer::new_from_core(core, 51866, 1500).unwrap();
    assert_eq!(t.language_token("yue"), Some(51865));
    assert_eq!(t.timestamp_begin, 50364);
    assert_eq!(t.timestamp_token(30.0), 51864);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test whisper_tokenizer`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/tokenizer/whisper.rs`:
```rust
use crate::{Error, Result};
use crate::tokenizer::tiktoken::CoreBpe;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    Translate,
}

const EOT: u32 = 50256;
const SOT: u32 = 50257;
const TRANSLATE: u32 = 50358;
const TRANSCRIBE: u32 = 50359;
const STARTOFLM: u32 = 50360;
const STARTOPREV: u32 = 50361;
const NOSPEECH: u32 = 50362;
const NOTIMESTAMPS: u32 = 50363;
const LANGUAGE_BASE: u32 = 50258;

/// Whisper's 99 languages, sorted by code (matches reference `LANGUAGES`).
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
    "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
    "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
    "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
    "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
    "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
    "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
    "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
];

pub struct TextTokenizer {
    pub core: CoreBpe,
    pub n_vocab: u32,
    pub n_audio_ctx: usize,
    pub timestamp_begin: u32,
    pub language_ids: HashMap<&'static str, u32>,
}

fn make_specials(n_vocab: u32, n_audio_ctx: usize) -> Vec<(String, u32)> {
    let mut v = vec![
        ("<|endoftext|>".to_string(), EOT),
        ("<|startoftranscript|>".to_string(), SOT),
        ("<|translate|>".to_string(), TRANSLATE),
        ("<|transcribe|>".to_string(), TRANSCRIBE),
        ("<|startoflm|>".to_string(), STARTOFLM),
        ("<|startofprev|>".to_string(), STARTOPREV),
        ("<|nospeech|>".to_string(), NOSPEECH),
        ("<|notimestamps|>".to_string(), NOTIMESTAMPS),
    ];
    for (i, code) in LANGUAGES.iter().enumerate() {
        v.push((format!("<|{code}|>"), LANGUAGE_BASE + i as u32));
    }
    // large-v3 (n_vocab == 51866): <|yue|> appended at the end
    if n_vocab == 51866 {
        v.push(("<|yue|>".to_string(), n_vocab - 1));
    }
    let timestamp_begin = n_vocab.saturating_sub(n_audio_ctx as u32).saturating_sub(1);
    v.push(("<|0.00|>".to_string(), timestamp_begin));
    v
}

impl TextTokenizer {
    pub fn new(bytes: &[u8], n_vocab: u32, n_audio_ctx: usize) -> Result<Self> {
        let specials = make_specials(n_vocab, n_audio_ctx);
        let core = CoreBpe::from_tiktoken_with_special(bytes, &specials)?;
        Self::new_from_core(core, n_vocab, n_audio_ctx)
    }

    pub fn new_from_core(core: CoreBpe, n_vocab: u32, n_audio_ctx: usize) -> Result<Self> {
        let timestamp_begin = n_vocab
            .checked_sub(n_audio_ctx as u32)
            .and_then(|x| x.checked_sub(1))
            .ok_or_else(|| Error::Tokenizer("n_vocab < n_audio_ctx+1".into()))?;
        let mut language_ids = HashMap::new();
        for (i, code) in LANGUAGES.iter().enumerate() {
            language_ids.insert(*code, LANGUAGE_BASE + i as u32);
        }
        if n_vocab == 51866 {
            language_ids.insert("yue", n_vocab - 1);
        }
        Ok(Self {
            core,
            n_vocab,
            n_audio_ctx,
            timestamp_begin,
            language_ids,
        })
    }

    pub fn sot(&self) -> u32 { SOT }
    pub fn eot(&self) -> u32 { EOT }
    pub fn translate(&self) -> u32 { TRANSLATE }
    pub fn transcribe(&self) -> u32 { TRANSCRIBE }
    pub fn startoflm(&self) -> u32 { STARTOFLM }
    pub fn startofprev(&self) -> u32 { STARTOPREV }
    pub fn nospeech(&self) -> u32 { NOSPEECH }
    pub fn notimestamps(&self) -> u32 { NOTIMESTAMPS }

    pub fn language_token(&self, code: &str) -> Option<u32> {
        self.language_ids.get(code).copied()
    }

    pub fn valid_language_tokens(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.language_ids.values().copied().collect();
        v.sort_unstable();
        v
    }

    pub fn timestamp_token(&self, time: f64) -> u32 {
        let steps = (time.round() / 0.02) as u32;
        self.timestamp_begin + steps
    }

    pub fn timestamp_tokens(&self, start: f64, end: f64) -> Vec<u32> {
        let a = self.timestamp_token(start);
        let b = self.timestamp_token(end);
        let mut out = Vec::new();
        let step = if b >= a { 1 } else { -1 };
        let mut t = a;
        loop {
            out.push(t);
            if t == b {
                break;
            }
            if step > 0 {
                t += 1;
            } else {
                t -= 1;
            }
        }
        out
    }

    pub fn sot_sequence(&self, lang: Option<&str>, task: Task, timestamps: bool) -> Result<Vec<u32>> {
        let mut seq = vec![self.sot()];
        if let Some(code) = lang {
            seq.push(self
                .language_token(code)
                .ok_or_else(|| Error::Tokenizer(format!("unknown language {code}")))?);
        }
        seq.push(match task {
            Task::Transcribe => self.transcribe(),
            Task::Translate => self.translate(),
        });
        if !timestamps {
            seq.push(self.notimestamps());
        }
        Ok(seq)
    }

    /// Return true if a token is in [timestamp_begin, n_vocab) (a timestamp token).
    pub fn is_timestamp(&self, token: u32) -> bool {
        token >= self.timestamp_begin && token < self.n_vocab
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.core.encode_ordinary(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.core.decode(tokens)
    }
}
```

Modify `crates/whisper-burn/src/tokenizer/mod.rs`:
```rust
pub mod tiktoken;
pub mod whisper;
```

Place the two tests from Step 1 in `crates/whisper-burn/tests/whisper_tokenizer.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test whisper_tokenizer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: add whisper tokenizer with special tokens, languages, timestamps"
```

---

### Task 7: STFT (torch `center=True` semantics)

**Files:**
- Create: `crates/whisper-burn/src/features/mod.rs`
- Create: `crates/whisper-burn/src/features/stft.rs`

**Interfaces:**
- Produces:
  - `pub struct StftConfig { pub n_fft: usize, pub hop_length: usize }` with `impl Default` (n_fft=400, hop_length=160)
  - `pub fn stft_power(samples: &[f32], config: &StftConfig) -> Vec<Vec<f32>>` — returns `[n_frames][n_fft/2+1]` of `|X|^2`, DFT (O(n_fft²) naive). Padding: `n_fft/2` zeros on both sides (center=True). `n_frames = samples.len() / hop_length + 1`.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/src/features/stft.rs` unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Verified by hand: x=[1,1,1,1], n_fft=4, hop=2, Hann.
    fn hann(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()).collect()
    }

    #[test]
    fn golden_small_frame_power() {
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let cfg = StftConfig { n_fft: 4, hop_length: 2 };
        let power = stft_power(&x, &cfg);
        // frames = 4/2 + 1 = 3
        assert_eq!(power.len(), 3);
        assert_eq!(power[0].len(), 3); // n_fft/2+1
        let w = hann(4); // [0.0, 0.5, 1.0, 0.5]
        assert_eq!(w, vec![0.0, 0.5, 1.0, 0.5]);
        // frame 0 window over padded [0,0,1,1] -> x*w = [0,0,1,0.5]
        //  X0 = 1.5, |X1|^2=1.25, |X2|^2=0.25
        assert!((power[0][0] - 2.25).abs() < 1e-4);
        assert!((power[0][1] - 1.25).abs() < 1e-4);
        assert!((power[0][2] - 0.25).abs() < 1e-4);
        // frame 1 over padded [1,1,1,1] -> x*w=[0,0.5,1,0.5]
        assert!((power[1][0] - 4.0).abs() < 1e-4);
        assert!((power[1][1] - 1.0).abs() < 1e-4);
        assert!((power[1][2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn sine_tone_peaks_at_bin() {
        let sr = 16000.0;
        let f = 400.0; // 400 Hz -> bin k = f * n_fft / sr = 10
        let n = 4000usize;
        let cfg = StftConfig { n_fft: 400, hop_length: 160 };
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin())
            .collect();
        let power = stft_power(&x, &cfg);
        let mid = power.len() / 2;
        let peak = power[mid]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 10);
        // frames = 4000/160 + 1 = 26
        assert_eq!(power.len(), 26);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --lib stft`
Expected: FAIL — `stft_power` missing.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/features/mod.rs`:
```rust
pub mod mel;
pub mod stft;
pub mod window;
```
(create `mel.rs`/`window.rs` as empty placeholders so the crate compiles; they're filled in Task 8)

`crates/whisper-burn/src/features/stft.rs`:
```rust
#[derive(Clone, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self { n_fft: 400, hop_length: 160 }
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

fn dft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_freq = n / 2 + 1;
    let pi2 = 2.0 * std::f32::consts::PI;
    (0..n_freq)
        .map(|k| {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (m, &x) in frame.iter().enumerate() {
                let ang = -pi2 * k as f32 * m as f32 / n as f32;
                re += x * ang.cos();
                im += x * ang.sin();
            }
            re * re + im * im
        })
        .collect()
}

pub fn stft_power(samples: &[f32], config: &StftConfig) -> Vec<Vec<f32>> {
    let pad = config.n_fft / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.resize(pad, 0.0);
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + 2 * pad, 0.0);

    let n_frames = samples.len() / config.hop_length + 1;
    let window = hann(config.n_fft);

    let mut out = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let start = t * config.hop_length;
        let mut frame = Vec::with_capacity(config.n_fft);
        for (b, &w) in window.iter().enumerate() {
            frame.push(padded[start + b] * w);
        }
        out.push(dft_power(&frame));
    }
    out
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --lib stft`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: add centered DFT STFT producing power spectra"
```

---

### Task 8: Mel filterbank assets + log-mel transform

**Files:**
- Create: `scripts/extract_mel_filters.py` (one-time dev script, committed)
- Create: `crates/whisper-burn/assets/mel_filters_80.bin`, `mel_filters_128.bin` (generated by the script, committed)
- Create: `crates/whisper-burn/src/features/mel.rs`
- Fill: `crates/whisper-burn/src/features/window.rs`
- Modify: `crates/whisper-burn/src/lib.rs` (register `assets` path? no — assets are embedded via `include_bytes!` with relative path); `crates/whisper-burn/src/features/mod.rs`

**Interfaces:**
- Produces:
  - `pub const N_SAMPLES: usize = 480_000; pub const N_FRAMES: usize = 3_000;`
  - `pub fn split_chunks(samples: &[f32]) -> Vec<Vec<f32>>` — each chunk exactly `N_SAMPLES` samples; last chunk zero-padded. Returns a `Vec` of chunks; also returns chunk sample-counts? Use `pub fn split_chunks(samples: &[f32]) -> Vec<(Vec<f32>, usize)>` where second element is the true sample count (for accurate timestamps).
  - `pub struct FeatureExtractor { n_mels: usize, n_fft: usize, hop_length: usize, filters: Vec<f32> /* n_mels * (n_fft/2+1) */, sample_rate: u32 }`
  - `impl FeatureExtractor { pub fn new(n_mels: usize, sample_rate: u32) -> Result<Self>; pub fn log_mel(&self, samples: &[f32]) -> Result<Vec<f32>> }` — returns `n_mels * n_frames` row-major `[n_mels][n_frames]`, where `n_frames = samples.len()/hop + 1`.
  - Baked assets persisted as little-endian f32 row-major `[n_mels][n_fft/2+1]`.
  - log-mel math: `power = stft**2`, `mel[m,t] = sum_b filters[m][b] * power[t][b]`, `log = log10(max(mel, 1e-10))`, `log = max(log, max(log) - 8)`, `out = (log + 4)/4`.

- [ ] **Step 1: Write the failing test**

`scripts/extract_mel_filters.py`:
```python
"""One-time dev tool: extract Whisper's mel filters to Rust assets.

Usage:  python scripts/extract_mel_filters.py
Requires numpy. Reads the official mel_filters.npz (download it from the
openai/whisper repo root: whisper/assets/mel_filters.npz) and writes
little-endian f32 row-major arrays, one filter per row.
"""
import numpy as np, pathlib, urllib.request

URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
outdir = pathlib.Path("crates/whisper-burn/assets")
outdir.mkdir(parents=True, exist_ok=True)

tmp = pathlib.Path("crates/whisper-burn/target/mel_filters.npz")
if not tmp.exists():
    tmp.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, tmp)

data = np.load(tmp)
for n in (80, 128):
    arr = data[f"mel_{n}"]          # shape (n_fft//2+1, n_mels) in the npz
    arr = np.ascontiguousarray(arr.T, dtype=np.float32)  # (n_mels, 201)
    arr.tofile(outdir / f"mel_filters_{n}.bin")
    print(n, arr.shape)
# Print a provenance checksum of the first row for later cement
print("first row of mel_80:", data["mel_80"][0, :6])
```

Run the script now (Step 1a; requires network + numpy):
`python scripts/extract_mel_filters.py`
Expected: prints `80 (80, 201)` / `128 (128, 201)` and the files exist. Commit the generated `.bin` files.

`crates/whisper-burn/tests/mel.rs`:
```rust
use whisper_burn::features::mel::FeatureExtractor;
use whisper_burn::features::window::{split_chunks, N_FRAMES, N_SAMPLES};

const FILTERS_80: &[u8] = include_bytes!("../assets/mel_filters_80.bin");
const FILTERS_128: &[u8] = include_bytes!("../assets/mel_filters_128.bin");

#[test]
fn assets_have_expected_shape() {
    assert_eq!(FILTERS_80.len(), 80 * 201 * 4);
    assert_eq!(FILTERS_128.len(), 128 * 201 * 4);
}

#[test]
fn chunking_pads_to_30s() {
    let s = vec![0.0f32; N_SAMPLES + 1000];
    let chunks = split_chunks(&s);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].0.len(), N_SAMPLES);
    assert_eq!(chunks[0].1, N_SAMPLES);
    assert_eq!(chunks[1].0.len(), N_SAMPLES);
    assert_eq!(chunks[1].1, 1000);
    assert!(chunks[1].0[N_SAMPLES - 1] == 0.0);
    let _ = N_FRAMES;
}

#[test]
fn log_mel_shape_and_math() {
    let fe = FeatureExtractor::new(80, 16000).unwrap();
    // 1 second of silence -> (3000 frames from 480k? use 16000 samples -> 101 frames)
    let s = vec![0.0f32; 16000];
    let m = fe.log_mel(&s).unwrap();
    let n_frames = 16000 / 160 + 1;
    assert_eq!(m.len(), 80 * n_frames);
    // all zeros audio => zero power => log10(0)->-inf clamped to 1e-10 -> -10 -> max-8 -> (x+4)/4
    assert!(m.iter().all(|&v| v > -30.0 && v < 30.0));
    // constant positive DC power gives a finite, computable mel value
    let dc = vec![1.0f32; 16000];
    let m2 = fe.log_mel(&dc).unwrap();
    assert!(m2.iter().all(|&v| v.is_finite()));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test mel`
Expected: FAIL — `FeatureExtractor` missing; assets missing.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/features/window.rs`:
```rust
pub const N_SAMPLES: usize = 480_000; // 30 s at 16 kHz
pub const N_FRAMES: usize = 3_000;    // 30 s in 10 ms frames

pub fn split_chunks(samples: &[f32]) -> Vec<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + N_SAMPLES).min(samples.len());
        let mut chunk = vec![0.0f32; N_SAMPLES];
        chunk[..end - start].copy_from_slice(&samples[start..end]);
        out.push((chunk, end - start));
        start = end;
    }
    out
}
```

`crates/whisper-burn/src/features/mel.rs`:
```rust
use crate::{Error, Result};
use super::stft::{stft_power, StftConfig};

pub struct FeatureExtractor {
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    filters: Vec<f32>,
    sample_rate: u32,
}

fn load_filters(n_mels: usize) -> Result<Vec<f32>> {
    let bytes = match n_mels {
        80 => include_bytes!("../../assets/mel_filters_80.bin").as_slice(),
        128 => include_bytes!("../../assets/mel_filters_128.bin").as_slice(),
        other => return Err(Error::UnsupportedFormat(format!("n_mels={other}"))),
    };
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl FeatureExtractor {
    pub fn new(n_mels: usize, sample_rate: u32) -> Result<Self> {
        let filters = load_filters(n_mels)?;
        Ok(Self {
            n_mels,
            n_fft: 400,
            hop_length: 160,
            filters,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn log_mel(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let cfg = StftConfig { n_fft: self.n_fft, hop_length: self.hop_length };
        let power = stft_power(samples, &cfg);
        let n_freq = self.n_fft / 2 + 1;
        let n_frames = power.len();
        let mut mel = vec![0.0f32; self.n_mels * n_frames];
        // [m][t] = sum_b filters[m][b] * power[t][b]
        for m in 0..self.n_mels {
            for t in 0..n_frames {
                let mut acc = 0.0f32;
                for b in 0..n_freq {
                    acc += self.filters[m * n_freq + b] * power[t][b];
                }
                mel[m * n_frames + t] = acc;
            }
        }
        // log10 with floor, then max-8 clamp, then (x+4)/4
        for v in mel.iter_mut() {
            let x = (*v).max(1e-10).log10();
            *v = x;
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = max - 8.0;
        for v in mel.iter_mut() {
            *v = ((*v).max(floor) + 4.0) / 4.0;
        }
        Ok(mel)
    }
}
```
Note: `log_mel` only uses `sample_rate` for `log_mel` semantics check (mel assumes 16 kHz); resampling to 16 kHz is handled in `audio/resample.rs` before calling.

`crates/whisper-burn/src/features/mod.rs`:
```rust
pub mod mel;
pub mod stft;
pub mod window;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test mel`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ crates/whisper-burn src/ 
git commit -m "feat: bake mel filterbanks and add log-mel feature extractor + 30s windowing"
```

---

### Task 9: Audio decode (symphonia)

**Files:**
- Create: `crates/whisper-burn/src/audio/mod.rs`
- Create: `crates/whisper-burn/src/audio/decode.rs` (behind `#[cfg(feature = "audio")]`)
- Modify: `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces (feature `audio`):
  - `pub fn decode_to_mono_f32(path: impl AsRef<std::path::Path>) -> Result<(Vec<f32>, u32)>` — returns interleaved audio converted to mono f32 and the source sample rate. Any symphony-supported container/codec.
  - `pub struct Decoded { pub samples: Vec<f32>, pub sample_rate: u32 }`

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/audio.rs` (guarded with `#![cfg(feature = "audio")]`):

```rust
#![cfg(feature = "audio")]
use whisper_burn::audio::decode::decode_to_mono_f32;
use std::io::Write as _;

fn write_wav(path: &std::path::Path, samples: &[f32], sr: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    // minimal PCM16 WAV header
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut hdr = Vec::new();
    hdr.extend(b"RIFF");
    hdr.extend((36 + data_len).to_le_bytes());
    hdr.extend(b"WAVE");
    hdr.extend(b"fmt ");
    hdr.extend(16u32.to_le_bytes());
    hdr.extend(1u16.to_le_bytes()); // PCM
    hdr.extend(1u16.to_le_bytes()); // mono
    hdr.extend(sr.to_le_bytes());
    hdr.extend((sr * 2).to_le_bytes());
    hdr.extend(2u16.to_le_bytes()); // block align
    hdr.extend(16u16.to_le_bytes()); // bits
    hdr.extend(b"data");
    hdr.extend(data_len.to_le_bytes());
    f.write_all(&hdr).unwrap();
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

#[test]
fn decodes_wav_mono() {
    let dir = std::env::temp_dir();
    let path = dir.join("whisper_test_tone.wav");
    let sr = 16000u32;
    let samples: Vec<f32> = (0..1600)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();
    write_wav(&path, &samples, sr);
    let (out, got_sr) = decode_to_mono_f32(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(got_sr, sr);
    assert_eq!(out.len(), samples.len());
    for (a, b) in out.iter().zip(samples.iter()) {
        assert!((a - b).abs() < 0.02);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --features audio --test audio`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/audio/mod.rs`:
```rust
pub mod decode;
```

`crates/whisper-burn/src/audio/decode.rs`:
```rust
use crate::{Error, Result};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use std::io::{Read, Seek};

pub struct Decoded {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_to_mono_f32(path: impl AsRef<std::path::Path>) -> Result<Decoded> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Io { path: path.to_path_buf(), source: e })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let meta_opts = MetadataOptions::default();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &meta_opts)
        .map_err(|e| Error::AudioDecode(format!("probe {}: {e}", path.display())))?;

    let mut reader = probed.format;
    let track = reader
        .default_track()
        .ok_or_else(|| Error::AudioDecode("no default track".into()))?;
    let sample_rate = track.codec_params.sample_rate.ok_or_else(|| {
        Error::AudioDecode("unknown sample rate".into())
    })?;
    let channels = track.codec_params.channels.ok_or_else(|| {
        Error::AudioDecode("unknown channel layout".into())
    })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode(format!("codec: {e}")))?;

    let mut accum: Vec<Vec<f32>> = vec![Vec::new(); channels.count()];
    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::AudioDecode(format!("packet: {e}"))),
        };
        let mut buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(Error::AudioDecode(format!("decode: {e}"))),
        };
        let spec = *buf.spec();
        let mut sbuf: SampleBuffer<f32> = SampleBuffer::new(buf.capacity() as u64, spec);
        sbuf.copy_interleaved_ref(buf);
        let inter = sbuf.samples();
        let n_ch = spec.channels.count();
        for (i, &s) in inter.iter().enumerate() {
            accum[i % n_ch].push(s);
        }
        let _ = &mut buf;
    }

    let mut out = Vec::with_capacity(accum[0].len());
    let n_ch = accum.len();
    for i in 0..accum[0].len() {
        let sum: f32 = accum.iter().map(|c| c[i]).sum();
        out.push(sum / n_ch as f32);
    }
    Ok(Decoded { samples: out, sample_rate })
}
```

Modify `crates/whisper-burn/src/lib.rs`: add
```rust
#[cfg(feature = "audio")]
pub mod audio;
```

Note: the decoder is mono-only (downmixes channels by average) — matches a single-resample mono pipeline.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --features audio --test audio`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: decode audio files to mono f32 via symphonia"
```

---

### Task 10: Resampler to 16 kHz

**Files:**
- Create: `crates/whisper-burn/src/audio/resample.rs`
- Modify: `crates/whisper-burn/src/audio/mod.rs`

**Interfaces:**
- Produces (feature `audio`):
  - `pub fn resample_to_16k(samples: &[f32], in_sr: u32, out_sr: u32) -> Result<Vec<f32>>`
  - No-op when `in_sr == out_sr`; otherwise `rubato::SincFixedIn`.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/src/audio/resample.rs`:
```rust
#[cfg(all(test, feature = "audio"))]
mod tests {
    use super::*;

    #[test]
    fn noop_for_16k() {
        let s = vec![0.0f32; 100];
        let out = resample_to_16k(&s, 16000, 16000).unwrap();
        assert_eq!(out, s);
    }

    #[test]
    fn upsamples_8k_to_16k_length() {
        let sr = 8000u32;
        let f = 1000.0;
        let s: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect();
        let out = resample_to_16k(&s, sr, 16000).unwrap();
        // ~2x length
        assert!(out.len() as f32 / s.len() as f32 > 1.9);
        assert!(out.len() as f32 / s.len() as f32 < 2.1);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --features audio --lib resample`
Expected: FAIL — function missing.

- [ ] **Step 3: Write the implementation**

```rust
use crate::{Error, Result};

pub fn resample_to_16k(samples: &[f32], in_sr: u32, out_sr: u32) -> Result<Vec<f32>> {
    if in_sr == out_sr {
        return Ok(samples.to_vec());
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let params = rubato::SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::InterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        samples.len().max(1),
        1,
    )
    .map_err(|e| Error::Resample(e.to_string()))?;
    let wav_in: Vec<Vec<f32>> = vec![samples.to_vec()];
    let wav_out = resampler
        .process(&wav_in, None)
        .map_err(|e| Error::Resample(e.to_string()))?;
    Ok(wav_out.into_iter().next().unwrap_or_default())
}
```

Modify `crates/whisper-burn/src/audio/mod.rs`:
```rust
pub mod decode;
pub mod resample;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --features audio --lib resample`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: resample PCM to 16 kHz via rubato"
```

---

### Task 11: WeightMap + custom Linear / LayerNorm / GELU(erf)

**Files:**
- Create: `crates/whisper-burn/src/weights/loader.rs`
- Create: `crates/whisper-burn/src/model/mod.rs`
- Create: `crates/whisper-burn/src/model/ops.rs`
- Modify: `crates/whisper-burn/src/weights/mod.rs`, `crates/whisper-burn/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub struct WeightMap<B: Backend> { st: SafeTensors, bytes: Vec<u8>, device: B::Device, consumed: HashSet<String> }`
  - `impl<B: Backend> WeightMap<B> { new(st, bytes, device), take_1d(name, len), take_2d(name, [r,c]), take_3d(name, [a,b,c]), finish() -> Result<()>, is_consumed(name) }`
  - `pub fn erf(x: f32) -> f32`, `pub fn gelu_erf(x: Tensor<B,3>) -> Tensor<B,3>` (through backend float ops)
  - `#[derive(Module, Debug)] pub struct Linear<B: Backend> { weight: Param<Tensor<B,2>>, bias: Option<Param<Tensor<B,1>>> }` with `forward(&self, x: Tensor<B,3>) -> Tensor<B,3>` = `x @ Wᵀ + b`
  - `#[derive(Module, Debug)] pub struct LayerNorm<B: Backend> { weight: Param<Tensor<B,1>>, bias: Param<Tensor<B,1>> }` with `forward(&self, x) -> Tensor<B,3>`, eps `1e-5`
  - All model tests run on `NdarrayBackend<f32>`.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/loader.rs` (test infra builds a small safetensors buffer):
```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::tensor::{Data, Tensor};
use whisper_burn::weights::loader::WeightMap;
use whisper_burn::weights::safetensors::{read_safetensors};

type B = NdarrayBackend<f32>;

fn build_file(json: &str, payload: &[u8]) -> Vec<u8> {
    let n = json.len() as u64;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(payload);
    out
}

fn flat_f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn drains_and_validates_weights() {
    let json = r#"{"enc.w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"enc.b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
    let mut payload = flat_f32s(&[1.0, 2.0, 3.0, 4.0]);
    payload.extend(flat_f32s(&[0.5, -0.5]));
    let bytes = build_file(json, &payload);
    let st = read_safetensors(&bytes).unwrap();
    let device = NdarrayDevice::default();
    let mut wm = WeightMap::<B>::new(st, bytes, device);
    let w: Tensor<B, 2> = wm.take_2d("enc.w", [2, 2]).unwrap();
    let b: Tensor<B, 1> = wm.take_1d("enc.b", 2).unwrap();
    assert_eq!(w.into_data(), Data::from([[1.0, 2.0], [3.0, 4.0]]));
    assert_eq!(b.into_data(), Data::from([0.5, -0.5]));
    wm.finish().unwrap();
}

#[test]
fn rejects_leftover_and_wrong_shape() {
    let json = r#"{"enc.w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}} "#;
    let mut bytes = build_file(json, &flat_f32s(&[1.0, 2.0, 3.0, 4.0]));
    // leftover: consume nothing → finish must error
    let st = read_safetensors(&bytes).unwrap();
    let wm = WeightMap::<B>::new(st, bytes.clone(), NdarrayDevice::default());
    assert!(wm.finish().is_err()); // leftover enc.w
    // wrong shape on take
    let mut wm2 = WeightMap::<B>::new(read_safetensors(&bytes).unwrap(), bytes.clone(), NdarrayDevice::default());
    assert!(wm2.take_2d("enc.w", [3, 3]).is_err());
    assert!(wm2.take_2d("enc.w", [2, 2]).is_ok());
    // double-consumption rejected
    assert!(wm2.take_2d("enc.w", [2, 2]).is_err());
    wm2.finish().unwrap();
}
```

`crates/whisper-burn/tests/ops.rs`:
```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::tensor::{Data, Tensor};
use whisper_burn::model::ops::{gelu_erf, erf, Linear, LayerNorm};
use burn::module::Module;
use burn::tensor::backend::Backend;

type B = NdarrayBackend<f32>;

#[test]
fn erf_matches_reference_values() {
    for (x, exp) in [(0.0, 0.0), (0.5, 0.5204998778), (1.0, 0.8427007929)] {
        assert!((erf(x) - exp).abs() < 1e-5, "erf({x}) = {} (expected {exp})", erf(x));
    }
}

#[test]
fn gelu_reflects_erf_form() {
    // 0.5 * 1 * (1 + erf(1/sqrt2)) = 0.84134...
    let device = NdarrayDevice::default();
    let x: Tensor<B, 3> = Tensor::from_data(Data::from([[[1.0f32]]]), &device);
    let y = gelu_erf(x);
    let v = y.into_data().to_vec::<f32>().unwrap()[0];
    assert!((v - 0.84134).abs() < 1e-4);
}

#[test]
fn linear_forward_matches_manual() {
    let device = NdarrayDevice::default();
    let w: Tensor<B, 2> = Tensor::from_data(Data::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let b: Tensor<B, 1> = Tensor::from_data(Data::from([0.5f32, -0.5]), &device);
    let lin = Linear { weight: burn::nn::Param::from(w), bias: Some(burn::nn::Param::from(b)) };
    let x: Tensor<B, 3> = Tensor::from_data(Data::from([[[1.0f32, 1.0], [2.0, 0.0]]]), &device);
    let y = lin.forward(x);
    // row0: x@W^T = [1*1+1*3, 1*2+1*4] = [4,6] + b = [4.5, 5.5]
    // row1: [2*1+0*3, 2*2+0*4] = [2,4] + b = [2.5, 3.5]
    assert_eq!(
        y.into_data(),
        Data::from([[[4.5, 5.5], [2.5, 3.5]]])
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p whisper-burn --test loader --test ops`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/weights/loader.rs`:
```rust
use crate::weights::safetensors::{read_safetensors, slice_f32, SafeTensors};
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor};
use std::collections::HashSet;

pub struct WeightMap<B: Backend> {
    st: SafeTensors,
    bytes: Vec<u8>,
    device: B::Device,
    consumed: HashSet<String>,
}

impl<B: Backend> WeightMap<B> {
    pub fn new(st: SafeTensors, bytes: Vec<u8>, device: B::Device) -> Self {
        Self { st, bytes, device, consumed: HashSet::new() }
    }

    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.st.all_tensor_names()
    }

    pub fn is_consumed(&self, name: &str) -> bool {
        self.consumed.contains(name)
    }

    fn take_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.consumed.contains(name) {
            return Err(Error::MissingWeight(format!("{name} already consumed")));
        }
        let meta = self
            .st
            .by_name(name)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))?;
        let vals = slice_f32(meta, &self.bytes)?.to_vec();
        self.consumed.insert(name.to_string());
        Ok((vals, meta.shape.clone()))
    }

    pub fn take_1d(&mut self, name: &str, len: usize) -> Result<Tensor<B, 1>> {
        let (v, s) = self.take_f32(name)?;
        if s != vec![len] {
            return Err(Error::ShapeMismatch { name: name.into(), expected: vec![len], got: s });
        }
        Ok(Tensor::<B, 1>::from_data(Data::new(v, Shape::from([len])), &self.device))
    }

    pub fn take_2d(&mut self, name: &str, shape: [usize; 2]) -> Result<Tensor<B, 2>> {
        let (v, s) = self.take_f32(name)?;
        if s != shape.to_vec() {
            return Err(Error::ShapeMismatch { name: name.into(), expected: shape.to_vec(), got: s });
        }
        Ok(Tensor::<B, 2>::from_data(Data::new(v, Shape::from(shape)), &self.device))
    }

    pub fn take_3d(&mut self, name: &str, shape: [usize; 3]) -> Result<Tensor<B, 3>> {
        let (v, s) = self.take_f32(name)?;
        if s != shape.to_vec() {
            return Err(Error::ShapeMismatch { name: name.into(), expected: shape.to_vec(), got: s });
        }
        Ok(Tensor::<B, 3>::from_data(Data::new(v, Shape::from(shape)), &self.device))
    }

    pub fn finish(&self) -> Result<()> {
        let remaining: Vec<String> = self
            .st
            .all_tensor_names()
            .filter(|n| !self.consumed.contains(*n))
            .map(|n| n.to_string())
            .collect();
        if remaining.is_empty() {
            Ok(())
        } else {
            Err(Error::UnexpectedWeight(remaining.join(", ")))
        }
    }
}
```

`crates/whisper-burn/src/model/ops.rs`:
```rust
use burn::module::Module;
use burn::nn::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// A&S 7.1.26 erf approximation (max abs error ~1.5e-7 in f64; <1e-6 in f32).
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn gelu_erf<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    let sqrt2 = B::FloatElem::from_f64(std::f64::consts::SQRT_2);
    let one = B::FloatElem::from(1.0);
    let half = B::FloatElem::from(0.5);
    let x_ov = x.clone().div_scalar(sqrt2);
    // erf is elementwise; use tensor map through backend float
    let erf = x_ov.map(|v| erf(v));
    x.mul(erf.add_scalar(one)).mul_scalar(half)
}

#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> Linear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let w = self.weight.val();
        let mut y = x.matmul(w.transpose());
        if let Some(b) = &self.bias {
            y = y.add(b.val().unsqueeze().unsqueeze());
        }
        y
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(weight: Param<Tensor<B, 1>>, bias: Param<Tensor<B, 1>>) -> Self {
        Self { weight, bias, epsilon: 1e-5 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dim_last = burn::tensor::Dim::Last;
        let mean = x.clone().mean_dim(dim_last).unsqueeze_dims(2, 1);
        let xm = x.clone().sub(mean.clone());
        let var = xm.clone().mul(xm.clone()).mean_dim(dim_last).unsqueeze_dims(2, 1);
        let eps = B::FloatElem::from_f64(self.epsilon);
        let denom = var.add_scalar(eps).powf(B::FloatElem::from(-0.5));
        let norm = xm.mul(denom);
        norm.mul(self.weight.val().unsqueeze().unsqueeze())
            .add(self.bias.val().unsqueeze().unsqueeze())
    }
}
```
Note: `Tensor::map` requires `B::FloatElem: burn::tensor::Element`; ops are elementwise over f32. `from_f64` comes from `burn::tensor::ElementConversion`.

`crates/whisper-burn/src/weights/mod.rs` (append to existing `pub mod safetensors;` from Task 4):
```rust
pub mod safetensors;
pub mod loader;
```

`crates/whisper-burn/src/model/mod.rs`:
```rust
pub mod ops;
```

Modify `crates/whisper-burn/src/lib.rs`:
```rust
pub mod model;
pub mod weights;   // already added
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p whisper-burn --test loader --test ops`
Expected: PASS. If Burn's API differs (e.g. `mean_dim(Dim::Last)` shape/semantics), adjust ops.rs to keep these two golden tests green — they pin the math.

- [ ] **Step 5: Commit**

```bash
git add crates/whisper-burn
git commit -m "feat: WeightMap, erf-GELU, Linear, LayerNorm"
```

---

### Task 12: MultiHeadAttention (scaled dot-product, causal + KV cross)

**Files:**
- Create: `crates/whisper-burn/src/model/attention.rs`
- Modify: `crates/whisper-burn/src/model/mod.rs`

**Interface:**
- `pub struct MultiHeadAttention<B> { q: Linear<B>, k: Linear<B>, v: Linear<B>, out: Linear<B>, n_head: usize }` (Linear over last dim: `x @ Wᵀ + b`)
- `impl<B: Backend> MultiHeadAttention<B> { pub fn new(q,k,v,out, n_head) -> Self; pub fn forward(&self, x: Tensor<B,3>, xa: Option<Tensor<B,3>>, mask: Option<&Tensor<B,3>>) -> Tensor<B,3> }`
  - self-attn when `xa.is_none()` (q,k,v all from `x`), cross-attn when `Some(xa)` (k,v from `xa`).
  - q·kᵀ / √d_head softmax, mask additive (`mask` = −inf on forbidden cells), then ·v, concat heads, `out` project.
  - `mask` shape `[1, 1, q, k]` consumed via `.add()` broadcast; causal mask built by caller.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/attention.rs` — verify exact math on a single head with identity weights:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::nn::Param;
use burn::tensor::{Data, Tensor};
use whisper_burn::model::attention::MultiHeadAttention;
use whisper_burn::model::ops::Linear;

type B = NdarrayBackend<f32>;

// q=k=v=out all act as identity (weight=[I], bias=0) → attention degrades to causal-mean of x.
fn identity_linear(device: &NdarrayDevice) -> Linear<B> {
    // weight shape [out, in]
    let w: Tensor<B, 2> = Tensor::from_data(Data::from([[1.0f32, 0.0], [0.0, 1.0]]), device);
    let b: Tensor<B, 1> = Tensor::from_data(Data::from([0.0f32, 0.0]), device);
    Linear { weight: Param::from(w), bias: Some(Param::from(b)) }
}

#[test]
fn self_attention_matches_hand_computation() {
    let device = NdarrayDevice::default();
    let l = identity_linear(&device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    let x: Tensor<B, 3> = Tensor::from_data(Data::from([[[1.0f32, 0.0], [0.0, 2.0]]]), &device);
    // q=k=v=x ; d_head=2 ; qk^T = [[1,0],[0,4]] /sqrt(2) ; causal → softmax rows
    // row0: softmax([1, -inf])  = [1, 0]  → out  [1,0]
    // row1: softmax([~0 , 4/1.414]) = [0.06, 0.94] → .06*[1,0]+.94*[0,2] = [0.06, 1.88]
    let y = mha.forward(x, None, None);
    let d = y.into_data().to_vec::<f32>().unwrap();
    assert!((d[0] - 1.0).abs() < 1e-3);
    assert!(d[1].abs() < 1e-3);
    assert!((d[2] - 0.060).abs() < 5e-3);
    assert!((d[3] - 1.88).abs() < 5e-3);
}

#[test]
fn cross_attention_uses_xa() {
    let device = NdarrayDevice::default();
    let l = identity_linear(&device);
    let mha = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    let q: Tensor<B, 3> = Tensor::from_data(Data::from([[[1.0f32, 0.0]]]), &device);
    let kv: Tensor<B, 3> = Tensor::from_data(Data::from([[[2.0f32, 0.0], [0.0, 0.0]]]), &device);
    let y = mha.forward(q, Some(kv), None);
    let d = y.into_data().to_vec::<f32>().unwrap();
    // both KV rows identical attention? different: row0 [2,0], row1 [0,0]; scores [2,0]/√2
    // softmax ≈ [0.806, 0.194] → out ≈ .806*[2,0]+.194*[0,0] = [1.61, 0.0]
    assert!((d[0] - 1.61).abs() < 5e-2, "got {:?}", d);
    assert!(d[1].abs() < 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails**
`cargo test -p whisper-burn --test attention`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/model/attention.rs`:
```rust
use super::ops::Linear;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor};

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub out: Linear<B>,
    pub n_head: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(query: Linear<B>, key: Linear<B>, value: Linear<B>, out: Linear<B>, n_head: usize) -> Self {
        Self { query, key, value, out, n_head }
    }

    fn qkv(&self, x: &Tensor<B, 3>, xa: Option<&Tensor<B, 3>>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let xa = xa.unwrap_or(x);
        (self.query.forward(x.clone()), self.key.forward(xa.clone()), self.value.forward(xa.clone()))
    }

    pub fn forward(&self, x: Tensor<B, 3>, xa: Option<Tensor<B, 3>>, mask: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(&x, xa.as_ref());
        let [b, m, chan] = q.shape().dims();
        let k_len = k.shape().dims()[1];
        let head = chan / self.n_head;

        let q = q.reshape([b, m, self.n_head, head]).swap_dims(1, 2);       // b,h,m,d
        let k = k.reshape([b, k_len, self.n_head, head]).swap_dims(1, 2);   // b,h,l,d
        let v = v.reshape([b, k_len, self.n_head, head]).swap_dims(1, 2);

        // scores: b,h,m,l
        let scale = B::FloatElem::from_f64(1.0 / (head as f64).sqrt());
        let mut scores = q.matmul(k.transpose().mul_scalar(scale));  // q (b,h,m,d) @ (b,h,l,d)^T
        // NOTE: py does q*(k^T * 1/√d). We fold scale into k, matching whisper ordering exactly.
        if let Some(mk) = mask {
            scores = scores.add(mk.clone());
        }
        let attn = burn::tensor::activation::softmax(scores, 3);     // softmax over l
        let ctx = attn.matmul(v);                                    // b,h,m,d
        let ctx = ctx.swap_dims(1, 2).reshape([-1, m, chan]);        // b,m,chan
        self.out.forward(ctx)
    }
}
```
> Scaling convention: whisper multiplies `k` by `1/√d_head` **after** the Q·Kᵀ transpose in `scaled_dot_product_attention` (`.reshape(..., q_len, -1)`), applied as `k * 1/√d`. Our `k.transpose().mul_scalar(scale)` is exactly that. The d_head used is `q.size()//n_head` — verify vs. cross-attention where `q
.shape` chan is n_state but cross reads `xa` channel count; reference uses `q.size(-1)//n_head`. We keep our `chan/n_head` on `q` — matches reference.

`crates/whisper-burn/src/model/mod.rs`:
```rust
pub mod attention;
pub mod ops;
```

- [ ] **Step 4: Run to verify it passes**
`cargo test -p whisper-burn --test attention` → PASS.

- [ ] **Step 5: Commit**
```bash
git add crates/whisper-burn
git commit -m "feat: multi-head scaled dot-product attention with causal/cross"
```

---

### Task 13: ResidualAttentionBlock

**Files:**
- Create: `crates/whisper-burn/src/model/block.rs`
- Modify: `crates/whisper-burn/src/model/mod.rs`

**Interface:**
- `pub struct ResidualAttentionBlock<B> { attn: MultiHeadAttention<B>, attn_ln: LayerNorm<B>, mlp: Linear<B>, mlp_ln: LayerNorm<B>, mlp2: Linear<B>, ln: LayerNorm<B>, xa_attn: Option<MultiHeadAttention<B>>, xa_attn_ln: Option<LayerNorm<B>>, xa_ln: Option<LayerNorm<B>> }`
- decode order is py `forward`: `x = x + attn(ln(x)); x = x + xa_attn(xa_ln(xa)); x = x + mlp2(gelu(mlp(mlp_ln(x))))`. Encoder uses same (xa fields None).
- `pub fn forward(&self, x, xa, mask) -> Tensor<B,3>` (encoder mask `None`).

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/block.rs` — with identity Linears and LayerNorm initialized to identity (=1 weight, 0 bias), output should equal `gelu`-path of a known input:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::nn::Param;
use burn::tensor::{Data, Tensor};
use whisper_burn::model::block::ResidualAttentionBlock;
use whisper_burn::model::attention::MultiHeadAttention;
use whisper_burn::model::ops::{gelu_erf, Linear, LayerNorm};

type B = NdarrayBackend<f32>;

fn id_linear(device: &NdarrayDevice) -> Linear<B> {
    let w: Tensor<B, 2> = Tensor::from_data(Data::from([[1.0f32, 0.0], [0.0, 1.0]]), device);
    let b: Tensor<B, 1> = Tensor::from_data(Data::from([0.0f32, 0.0]), device);
    Linear { weight: Param::from(w), bias: Some(Param::from(b)) }
}

fn id_ln(device: &NdarrayDevice) -> LayerNorm<B> {
    let w: Tensor<B, 1> = Tensor::from_data(Data::from([1.0f32, 1.0]), device);
    let b: Tensor<B, 1> = Tensor::from_data(Data::from([0.0f32, 0.0]), device);
    LayerNorm::new(Param::from(w), Param::from(b))
}

#[test]
fn encoder_block_single_token_identity_path() {
    // Encoder block: single token, all Linears identity, LayerNorm identity
    // → residual, attention, mlp all pass x through unchanged; final = x + gelu(x).
    let device = NdarrayDevice::default();
    let l = id_linear(&device);
    let ln = id_ln(&device);
    let attn = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l.clone(), 1);
    let block = ResidualAttentionBlock {
        attn,
        attn_ln: ln.clone(),
        mlp: l.clone(),
        mlp_ln: ln.clone(),
        mlp2: l,
        ln: ln.clone(),
        xa_attn: None,
        xa_attn_ln: None,
        xa_ln: None,
    };
    let x: Tensor<B, 3> = Tensor::from_data(Data::from([[[0.5f32, -0.25]]]), &device);
    let y = block.forward(x.clone(), None, None);
    let exp: Vec<f32> = x.clone().add(gelu_erf(x)).into_data().to_vec::<f32>().unwrap();
    let got: Vec<f32> = y.into_data().to_vec::<f32>().unwrap();
    assert_eq!(got, exp);
}

#[test]
fn decoder_cross_attention_uses_kv_and_query_ln() {
    // Identity weights: q ← x (post xa_attn_ln), k = v = xa.
    // x = [[1,0]] attends only over first KV row → out = that KV row.
    let device = NdarrayDevice::default();
    let l = id_linear(&device);
    let ln = id_ln(&device);
    let attn = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l.clone(), 1);
    let xa_attn = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l, 1);
    let block = ResidualAttentionBlock {
        attn,
        attn_ln: ln.clone(),
        mlp: l.clone(),
        mlp_ln: ln.clone(),
        mlp2: l.clone(),
        ln: ln.clone(),
        xa_attn: Some(xa_attn),
        xa_attn_ln: Some(ln.clone()),
        xa_ln: None,
    };
    let x: Tensor<B, 3> = Tensor::from_data(Data::from([[[1.0f32, 0.0], [0.0, 0.0]]]), &device);
    let xa: Tensor<B, 3> = Tensor::from_data(Data::from([[[2.0f32, 0.0], [0.0, 0.0], [0.0, 0.0]]]), &device);
    let y = block.forward(x, Some(xa), None);
    let got: Vec<f32> = y.into_data().to_vec::<f32>().unwrap();
    let x = Tensor::<B, 3>::from_data(Data::from([[[1.0f32, 0.0], [0.0, 0.0]]]), &device);
    let exp: Vec<f32> = x.into_data().to_vec::<f32>().unwrap();
    for (a, b) in got.iter().zip(exp.iter()) {
        assert!((a - b).abs() < 1e-3, "got {got:?} expect {exp:?}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**
`cargo test -p whisper-burn --test block`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/model/block.rs`:
```rust
use super::attention::MultiHeadAttention;
use super::ops::{gelu_erf, LayerNorm, Linear};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct ResidualAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub attn_ln: LayerNorm<B>,
    pub mlp: Linear<B>,
    pub mlp_ln: LayerNorm<B>,
    pub mlp2: Linear<B>,
    pub ln: LayerNorm<B>,
    pub xa_attn: Option<MultiHeadAttention<B>>,
    pub xa_attn_ln: Option<LayerNorm<B>>,
    pub xa_ln: Option<Linear<B>>, // matches py: nn.Linear(n_state, n_state, bias=False)
}

impl<B: Backend> ResidualAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Option<Tensor<B, 3>>, mask: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        // py: x = x + self.attn(self.attn_ln(x))
        let att = self.attn.forward(self.attn_ln.forward(x.clone()), None, mask);
        let x = x.add(att);
        // py: x = x + self.xa_attn(self.xa_attn_ln(x), k=xa, v=xa)
        if let (Some(xa), Some(xa_attn), Some(xa_attn_ln)) =
            (xa, &self.xa_attn, &self.xa_attn_ln)
        {
            let q = xa_attn_ln.forward(x.clone());
            let xa_att = xa_attn.forward(q, Some(xa), None);
            let x = x.add(xa_att);
        }
        // py: x = x + self.mlp2(self.gelu(self.mlp(self.mlp_ln(x))))
        let m = self.mlp.forward(self.mlp_ln.forward(x.clone()));
        let act = gelu_erf(m);
        x.add(self.mlp2.forward(act))
    }
}
```
> **Py-fidelity note:** in `ResidualAttentionBlock.forward` `xa_attn_ln` is applied to the **query** `x`, and the KV comes from the raw `xa` (encoder output, un-normalized here — normalization happens in `Decoder.forward` via `xa_ln`). The `xa_ln` field **exists in the state dict** (`decoder.blocks.{i}.xa_ln.weight`, shape `[n_state, n_state]`) but is **never used in forward** — we still must consume it from the WeightMap (see Task 16) or `finish()` fails. `ln` is also unused in forward but present as a loaded weight.

- [ ] **Step 4: Run to verify it passes**
`cargo test -p whisper-burn --test block` → PASS.

- [ ] **Step 5: Commit**
```bash
git add crates/whisper-burn
git commit -m "feat: residual attention block with cross-attention option"
```

---

### Task 14: AudioEncoder (conv1d + blocks + positional)

**Files:**
- Create: `crates/whisper-burn/src/model/encoder.rs`
- Modify: `crates/whisper-burn/src/model/mod.rs`

**Interface:**
- `pub struct Conv1d<B: Backend> { weight: Param<Tensor<B,3>>, bias: Param<Tensor<B,1>> }` with `forward(&self, x: Tensor<B,3>, stride: usize) -> Tensor<B,3>`. Weight `[out, in, k]`.
  - Same-pad (pad = (k-1)/2 symmetric zeros), kernel matmuls per offset, strided output via reshape trick for stride 2.
- `pub struct Encoder<B: Backend> { conv1: Conv1d<B>, conv2: Conv1d<B>, positional_embedding: Param<Tensor<B,2>>, blocks: Vec<ResidualAttentionBlock<B>>, ln_post: LayerNorm<B> }`
  - `forward(&self, mel: Tensor<B,3>) -> Tensor<B,3>` where mel `[1, n_mels, 3000]`; output `[1, 1500, n_state]` = `ln_post( blocks( gelu(conv2(gelu(conv1(x)))) + pos ) )`. Follows whisper exactly (pos added after convs, before blocks).

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/encoder.rs`:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::nn::Param;
use burn::tensor::{Data, Tensor};
use whisper_burn::model::encoder::{Conv1d, Encoder};
use whisper_burn::model::ops::LayerNorm;

type B = NdarrayBackend<f32>;

/// Naive reference conv1d: x [C][L], w [out][C][k], symmetric pad, stride s.
fn naive_conv1d(x: &[Vec<f32>], w: &[Vec<Vec<f32>>], b: &[f32], stride: usize, pad: usize) -> Vec<Vec<f32>> {
    let c = x.len();
    let l = x[0].len();
    let (nout, k) = (w.len(), w[0][0].len());
    let lp = l + 2 * pad;
    let xp: Vec<Vec<f32>> = x.iter().map(|row| {
        let mut v = vec![0.0; pad];
        v.extend(row.iter().cloned());
        v.extend(vec![0.0; pad]);
        v
    }).collect();
    let n_windows = lp - k + 1;          // stride-1 windows
    let out_len = n_windows / stride;    // stride-2 keeps even starts
    let mut out = vec![vec![0.0f32; out_len]; nout];
    for o in 0..nout {
        for t in 0..out_len {
            let mut acc = b[o];
            for ci in 0..c {
                for kk in 0..k {
                    acc += xp[ci][t * stride + kk] * w[o][ci][kk];
                }
            }
            out[o][t] = acc;
        }
    }
    out
}

#[test]
fn conv1d_matches_naive_reference() {
    let device = NdarrayDevice::default();
    let xd = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    ];
    let wd = vec![
        vec![vec![1.0, 0.0, -1.0], vec![0.0, 1.0, 0.0], vec![1.0, 1.0, 1.0]],
        vec![vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
    ];
    let bd = vec![0.1, -0.2];
    let x: Tensor<B, 3> = Tensor::from_data(Data::new(xd.iter().flatten().copied().collect(), [1, 3, 8]), &device);
    let w: Tensor<B, 3> = Tensor::from_data(Data::new(wd.iter().flatten().flatten().copied().collect(), [2, 3, 3]), &device);
    let bias: Tensor<B, 1> = Tensor::from_data(Data::new(bd.clone(), [2]), &device);
    let conv = Conv1d { weight: Param::from(w), bias: Param::from(bias) };
    for stride in [1usize, 2usize] {
        let got = conv.forward(x.clone(), stride).into_data().to_vec::<f32>().unwrap();
        let exp: Vec<f32> = naive_conv1d(&xd, &wd, &bd, stride, 1).into_iter().flatten().collect();
        assert_eq!(got.len(), exp.len(), "stride {stride}");
        for (a, b) in got.iter().zip(exp.iter()) {
            assert!((a - b).abs() < 1e-4, "stride {stride}: got {got:?}\nexp {exp:?}");
        }
    }
}

/// Encoder with all-zero conv weights/biases, all-zero transformer Linears,
/// identity LayerNorms, positional embedding = ones.
fn zero_encoder(dims: &whisper_burn::config::ModelDimensions, device: &NdarrayDevice) -> Encoder<B> {
    let s = dims.n_audio_state;
    let z1: Tensor<B, 1> = Tensor::zeros(&[s], device);
    let ln = LayerNorm { weight: Param::from(Tensor::ones(&[s], device)), bias: Param::from(z1.clone()), epsilon: 1e-5 };
    let block = || {
        let l = whisper_burn::model::ops::Linear {
            weight: Param::from(Tensor::zeros(&[s, s], device)),
            bias: Some(Param::from(z1.clone())),
        };
        let attn = whisper_burn::model::attention::MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l.clone(), dims.n_head);
        whisper_burn::model::block::ResidualAttentionBlock {
            attn, attn_ln: ln.clone(), mlp: l.clone(), mlp_ln: ln.clone(), mlp2: l,
            ln: ln.clone(), xa_attn: None, xa_attn_ln: None, xa_ln: None,
        }
    };
    Encoder {
        conv1: Conv1d { weight: Param::from(Tensor::zeros(&[s, dims.n_mels, 3], device)), bias: Param::from(z1.clone()) },
        conv2: Conv1d { weight: Param::from(Tensor::zeros(&[s, s, 3], device)), bias: Param::from(z1.clone()) },
        positional_embedding: Param::from(Tensor::ones(&[1500, s], device)),
        blocks: (0..dims.n_audio_layer).map(|_| block()).collect(),
        ln_post: ln,
    }
}

#[test]
fn zero_weight_encoder_out_is_all_ones() {
    use whisper_burn::config::{ModelDimensions, ModelSize};
    let dims = ModelSize::Tiny.dimensions();
    let device = NdarrayDevice::default();
    let enc = zero_encoder(&dims, &device);
    let mel: Tensor<B, 3> = Tensor::zeros(&[1, dims.n_mels, 3000], &device);
    let out = enc.forward(mel);
    assert_eq!(out.shape().dims(), &[1, 1500, dims.n_audio_state]);
    let v = out.into_data().to_vec::<f32>().unwrap();
    // zero convs → 0; +ones pos → 1; zero Linears in blocks → nothing added; LN identity → 1
    for x in v.iter() {
        assert!((x - 1.0).abs() < 1e-4, "got {x}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**
`cargo test -p whisper-burn --test encoder`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/model/encoder.rs`:
```rust
use crate::model::block::ResidualAttentionBlock;
use crate::model::ops::{gelu_erf, LayerNorm, Linear};
use burn::module::Module;
use burn::nn::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor};
use std::sync::Arc;

#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Conv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>, stride: usize) -> Tensor<B, 3> {
        let [b, c, l] = x.shape().dims();
        let [out, c2, k] = self.weight.val().shape().dims();
        assert_eq!(c, c2);
        let pad = (k - 1) / 2;
        // zero-pad symmetric
        let left: Tensor<B, 3> = Tensor::zeros(&[b, c, pad], &x.device());
        let right: Tensor<B, 3> = Tensor::zeros(&[b, c, pad], &x.device());
        let xp = Tensor::cat(vec![left, x, right], 2); // [b, c, l + 2p]
        let xp = xp.swap_dims(1, 2); // [b, L+2p, c]
        let lp = l + 2 * pad;
        let out_len = lp - k + 1;
        let mut acc: Option<Tensor<B, 3>> = None;
        for kk in 0..k {
            let w_k = self.weight.val().slice([0..out, 0..c, kk..kk + 1]).reshape([out, c]);
            let x_k = xp.clone().slice([0..b, kk..kk + out_len, 0..c]); // [b, out_len, c]
            let contrib = x_k.matmul(w_k.transpose()); // [b, out_len, out]
            acc = Some(match acc {
                Some(a) => a.add(contrib),
                None => contrib,
            });
        }
        let mut y = acc.unwrap().swap_dims(1, 2); // [b, out, out_len]
        if stride == 2 {
            // keep even positions: reshape [b,out,out_len/2,2] take [...,0]
            let half = out_len / 2;
            y = y.reshape([b, out, half, 2]);
            y = y.slice([0..b, 0..out, 0..half, 0..1]).squeeze(3);
        } else {
            assert_eq!(stride, 1);
        }
        let b1 = self.bias.val().unsqueeze().unsqueeze();
        y.add(b1)
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln_post: LayerNorm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = gelu_erf(self.conv1.forward(x, 1));       // [1, n_mels, 3000]
        x = gelu_erf(self.conv2.forward(x, 2));               // [1, n_state, 1500]
        x = x.swap_dims(1, 2);                                 // [1, 1500, n_state]
        x = x.add(self.positional_embedding.val().unsqueeze());
        for block in &self.blocks {
            x = block.forward(x, None, None);
        }
        self.ln_post.forward(x)
    }
}
```

`crates/whisper-burn/src/model/mod.rs`:
```rust
pub mod attention;
pub mod block;
pub mod encoder;
pub mod ops;
```

- [ ] **Step 4: Run to verify it passes**
`cargo test -p whisper-burn --test encoder` → PASS.

- [ ] **Step 5: Commit**
```bash
git add crates/whisper-burn
git commit -m "feat: conv1d + audio encoder from mel to token features"
```

---

### Task 15: TextDecoder (tied embeddings, causal mask, logits)

**Files:**
- Create: `crates/whisper-burn/src/model/decoder.rs`
- Modify: `crates/whisper-burn/src/model/mod.rs`

**Interface:**
- `pub struct Decoder<B> { token_embedding: Param<Tensor<B,2>> /* [n_vocab, n_state] */, positional_embedding: Param<Tensor<B,2>> /* [n_text_ctx, n_state] */, blocks: Vec<ResidualAttentionBlock<B>>, ln: LayerNorm<B> }`
- `forward(&self, tokens: Tensor<B,1,Int>, xa: Tensor<B,3>) -> Tensor<B,2>`:
  - embed `[seq, n_state]`, add `positional_embedding[0..seq]`, unsqueeze batch, run blocks with `xa` and causal mask, `ln`, then `logits = x @ token_embeddingᵀ` (tied, whisper uses `token_embedding.weightᵀ`).
  - Causal mask along `[-I]` above diagonal, shape `[1, 1, seq, seq]`.
  - whisper uses `mask = torch.full((n, n), -inf); mask.triu_(1)`.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/decoder.rs`:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::nn::Param;
use burn::tensor::{Data, Tensor};
use whisper_burn::model::decoder::Decoder;

type B = NdarrayBackend<f32>;

#[test]
fn decoder_zero_weights_matches_tied_manual() {
    use whisper_burn::config::{ModelDimensions, ModelSize};
    use whisper_burn::model::ops::LayerNorm;

    let dims = ModelSize::Tiny.dimensions(); // n_vocab, n_state, n_text_layer
    let device = NdarrayDevice::default();
    let s = dims.n_text_state;
    // token_embedding = identity on first s vocab rows (padded to n_vocab with zeros)
    let mut te_val: Vec<f32> = vec![0.0; dims.n_vocab * s];
    for i in 0..s {
        te_val[i * s + i] = 1.0;
    }
    let te: Tensor<B, 2> = Tensor::from_data(Data::new(te_val, [dims.n_vocab, s]), &device);
    let z1: Tensor<B, 1> = Tensor::zeros(&[s], device);
    let ln = LayerNorm { weight: Param::from(Tensor::ones(&[s], device)), bias: Param::from(z1.clone()), epsilon: 1e-5 };
    let block = || {
        use whisper_burn::model::attention::MultiHeadAttention;
        use whisper_burn::model::block::ResidualAttentionBlock;
        use whisper_burn::model::ops::Linear;
        let l = Linear { weight: Param::from(Tensor::zeros(&[s, s], device)), bias: Some(Param::from(z1.clone())) };
        let attn = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l.clone(), dims.n_head);
        let l2 = Linear { weight: Param::from(Tensor::zeros(&[s, s], device)), bias: Some(Param::from(z1.clone())) };
        let xa_attn = MultiHeadAttention::new(l.clone(), l.clone(), l.clone(), l2, dims.n_head);
        ResidualAttentionBlock { attn, attn_ln: ln.clone(), mlp: l.clone(), mlp_ln: ln.clone(), mlp2: l, ln: ln.clone(), xa_attn: Some(xa_attn), xa_attn_ln: Some(ln.clone()), xa_ln: None }
    };
    let dec = Decoder {
        token_embedding: Param::from(te.clone()),
        positional_embedding: Param::from(Tensor::zeros(&[dims.n_text_ctx, s], &device)),
        blocks: (0..dims.n_text_layer).map(|_| block()).collect(),
        ln,
    };
    // tokens [1, 2] → embed rows 0,1 = [e0, e1]; pos zero; zero attention → unchanged; ln identity
    let tokens: Tensor<B, 1, burn::tensor::Int> = Tensor::from_data(Data::from([0u32, 1u32]), &device).int();
    let xa: Tensor<B, 3> = Tensor::zeros(&[1, dims.n_audio_ctx, s], &device);
    let logits = dec.forward(tokens, xa);
    let v = logits.into_data().to_vec::<f32>().unwrap();
    // row0 = te[0] = [1,0,0,...], row1 = te[1] = [0,1,0,...]
    assert!((v[0] - 1.0).abs() < 1e-4, "got {v:?}");
    assert!(v[1].abs() < 1e-4);
    assert!(v[s].abs() < 1e-4);
    assert!((v[s + 1] - 1.0).abs() < 1e-4, "got {v:?}");
}

#[test]
fn causal_mask_shape_and_triangular() {
    use whisper_burn::model::decoder::causal_mask;
    use burn::backend::ndarray::NdarrayDevice;
    let device = NdarrayDevice::default();
    let m = causal_mask::<B>(3, &device);
    let d = m.into_data().to_vec::<f32>().unwrap();
    // strictly-lower-triangular entries must be 0 (allowed), upper/incl-diagonal handled by caller semantics:
    // whisper expects -inf on upper triangle. We assert our convention here:
    let expect = [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY,
                  0.0, 0.0, f32::NEG_INFINITY,
                  0.0, 0.0, 0.0];
    for (a, b) in d.iter().zip(expect.iter()) {
        assert_eq!(a.is_infinite(), b.is_infinite(), "got {d:?}");
    }
}
```

> Author note: verify `causal_mask` convention (lower-triangle zero, upper `-inf`) against the safe `apply_mask` in Task 16 — masking acts on `logits[:, :, :, :]` with `fill(-inf)` above the diagonal. If softmax of `-inf` on f32 wgpu returns non-zero, continue using `f32::NEG_INFINITY` (Burn's default mask), but see Task 16 note on NaN-safe masks.

- [ ] **Step 2: Run to verify it fails**
`cargo test -p whisper-burn --test decoder`
Expected: FAIL.

- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/model/decoder.rs`:
```rust
use crate::model::block::ResidualAttentionBlock;
use crate::model::ops::{LayerNorm};
use burn::module::Module;
use burn::nn::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor};

pub fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 4> {
    let n = n as i64;
    let mut m = vec![0.0f32; (n * n) as usize];
    for r in 0..n {
        for c in 0..n {
            if c > r { m[(r * n + c) as usize] = f32::NEG_INFINITY; }
        }
    }
    Tensor::from_data(Data::new(m, Shape::from([1, 1, n as usize, n as usize])), device)
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub token_embedding: Param<Tensor<B, 2>>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln: LayerNorm<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, tokens: Tensor<B, 1, burn::tensor::Int>, xa: Tensor<B, 3>) -> Tensor<B, 2> {
        let seq = tokens.shape().dims()[0];
        let n_state = self.token_embedding.val().shape().dims()[1];
        let emb = self.token_embedding.val().select(0, tokens);      // [seq, n_state]
        let pos = self.positional_embedding
            .val()
            .slice([0..seq, 0..n_state]);                            // [seq, n_state]
        let mut x = emb.add(pos).unsqueeze();                        // [1, seq, n_state]
        let mask = causal_mask::<B>(seq, &xa.device());
        for block in &self.blocks {
            x = block.forward(x, Some(xa.clone()), Some(&mask));
        }
        let x = self.ln.forward(x);                                  // [1, seq, n_state]
        let w = self.token_embedding.val().transpose();              // [n_state, n_vocab]
        x.squeeze(0).matmul(w)                                       // [seq, n_vocab]
    }
}
```

`crates/whisper-burn/src/model/mod.rs`:
```rust
pub mod attention;
pub mod block;
pub mod decoder;
pub mod encoder;
pub mod ops;
```
> Test note: `attn.clone_hack()` is fictional — clone the `MultiHeadAttention` by rebuilding 4 `Linear`s. Because blocks are built with zero weights that’s trivial; adjust the test accordingly (construct `Linear`s twice).

- [ ] **Step 4: Run to verify it passes**
`cargo test -p whisper-burn --test decoder` → PASS.

- [ ] **Step 5: Commit**
```bash
git add crates/whisper-burn
git commit -m "feat: text decoder with tied logits and causal mask"
```

---

### Task 16: Whisper model assembly + strict WeightMap drain

**Files:**
- Create: `crates/whisper-burn/src/model/whisper.rs`
- Modify: `crates/whisper-burn/src/model/mod.rs`

**Interface:**
- `pub struct Whisper<B: Backend> { pub dims: ModelDimensions, pub encoder: Encoder<B>, pub decoder: Decoder<B> }`
- `impl<B> Whisper<B> { pub fn from_weights(dims, wm: &mut WeightMap<B>, is_multilingual, n_text_ctx) -> Result<Self>; pub fn forward_encoder(&self, mel: Tensor<B,3>) -> Tensor<B,3>; pub fn forward_decoder(&self, tokens: Tensor<B,1,Int>, xa: &Tensor<B,3>) -> Tensor<B,2> }`
- **Weight naming convention** (HF/whisper keys, all must be consumed before `wm.finish()`):
  - encoder: `encoder.conv1.weight` [n_state, n_mels, 3], `encoder.conv1.bias` [n_state];
    `encoder.conv2.weight` [n_state, n_state, 3], `encoder.conv2.bias`;
    `encoder.positional_embedding` [1500, n_state];
    `encoder.blocks.{i}.` + `attn.query/k/v/out.weight+bias`, `attn_ln.weight+bias`, `mlp.weight+bias`, `mlp_ln.weight+bias`, `mlp2.weight+bias`, `ln.weight+bias`;
    `encoder.ln_post.weight+bias`.
  - decoder: `decoder.token_embedding.weight` [n_vocab, n_state]; `decoder.positional_embedding` [n_text_ctx, n_state];
    `decoder.blocks.{i}.` same as encoder plus `xa_attn.query/k/v/out.weight+bias`, `xa_attn_ln.weight+bias`, **`xa_ln.weight` [n_state, n_state, bias-less]** (present in state dict, unused in forward — must still be consumed);
    `decoder.ln.weight+bias`.
- Linear weight shape = `[out, in]` (`attn.query` [n_state, n_state], `mlp` [n_mlp, n_state], `mlp2` [n_state, n_mlp], `xa_ln` [n_state, n_state]).
- Conv2 has no bias? No — whisper `conv2` HAS a bias. Both convs have bias.
- `from_weights` must return `Err(Error::MissingWeight(...))` listing the first missing/unexpected weight, and callers run `wm.finish()` afterwards so *every* key is either loaded or errors.

- [ ] **Step 1: Write the failing test**

`crates/whisper-burn/tests/whisper.rs` — builds a full `Whisper` from a synthetic `SafeTensors` for tiny dims, checks a shape + that `finish()` passes, and that a missing weight errors:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use burn::tensor::{Data, Tensor};
use whisper_burn::config::{ModelSize, ModelDimensions};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::weights::loader::WeightMap;

type B = NdarrayBackend<f32>;
```
> Note the test must synthesize a complete weight set for `tiny` (matching the naming above). That's ~200 tensors; generate them programmatically in the test: iterate layers i, for each (prefix, list of (name, [r,c])) produce zeros/ones with the right sizes via a helper `push(json: &mut String, payload: &mut Vec<u8>, name, shape, fill)`. Use `1.0` fill so `forward_encoder` on a zero mel still returns finite numbers (verify shapes only; don't assert exact math here).

- [ ] **Step 2: Run to verify it fails** — `cargo test -p whisper-burn --test whisper`
- [ ] **Step 3: Write the implementation**

`crates/whisper-burn/src/model/whisper.rs`:
```rust
use crate::config::ModelDimensions;
use crate::model::attention::MultiHeadAttention;
use crate::model::block::ResidualAttentionBlock;
use crate::model::decoder::Decoder;
use crate::model::encoder::{Conv1d, Encoder};
use crate::model::ops::{LayerNorm, Linear};
use crate::weights::loader::WeightMap;
use crate::{Error, Result};
use burn::module::Module;
use burn::nn::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor};

fn ln<B: Backend>(wm: &mut WeightMap<B>, prefix: &str, n: usize) -> Result<LayerNorm<B>> {
    Ok(LayerNorm {
        weight: Param::from(wm.take_1d(&format!("{prefix}.weight"), n)?),
        bias: Param::from(wm.take_1d(&format!("{prefix}.bias"), n)?),
        epsilon: 1e-5,
    })
}

fn linear<B: Backend>(wm: &mut WeightMap<B>, prefix: &str, out: usize, nin: usize) -> Result<Linear<B>> {
    Ok(Linear {
        weight: Param::from(wm.take_2d(&format!("{prefix}.weight"), [out, nin])?),
        bias: Some(Param::from(wm.take_1d(&format!("{prefix}.bias"), out)?)),
    })
}

fn mha<B: Backend>(wm: &mut WeightMap<B>, prefix: &str, n_state: usize, n_head: usize) -> Result<MultiHeadAttention<B>> {
    let q = linear(wm, &format!("{prefix}.query"), n_state, n_state)?;
    let k = linear(wm, &format!("{prefix}.key"), n_state, n_state)?;
    let v = linear(wm, &format!("{prefix}.value"), n_state, n_state)?;
    let o = linear(wm, &format!("{prefix}.out"), n_state, n_state)?;
    Ok(MultiHeadAttention::new(q, k, v, o, n_head))
}

fn block<B: Backend>(wm: &mut WeightMap<B>, base: &str, n_state: usize, n_mlp: usize, n_head: usize, cross: bool) -> Result<ResidualAttentionBlock<B>> {
    let attn = mha(wm, &format!("{base}.attn"), n_state, n_head)?;
    let attn_ln = ln(wm, &format!("{base}.attn_ln"), n_state)?;
    let mlp = linear(wm, &format!("{base}.mlp"), n_mlp, n_state)?;
    let mlp_ln = ln(wm, &format!("{base}.mlp_ln"), n_state)?;
    let mlp2 = linear(wm, &format!("{base}.mlp2"), n_state, n_mlp)?;
    let lnb = ln(wm, &format!("{base}.ln"), n_state)?;
    let mut xa_attn = None;
    let mut xa_attn_ln = None;
    let xa_ln = if cross {
        xa_attn = Some(mha(wm, &format!("{base}.xa_attn"), n_state, n_head)?);
        xa_attn_ln = Some(ln(wm, &format!("{base}.xa_attn_ln"), n_state)?);
        // present in state dict, unused in forward: must consume to satisfy drain
        Some(Linear { weight: Param::from(wm.take_2d(&format!("{base}.xa_ln.weight"), [n_state, n_state])?), bias: None })
    } else {
        None
    };
    Ok(ResidualAttentionBlock { attn, attn_ln, mlp, mlp_ln, mlp2, ln: lnb, xa_attn, xa_attn_ln, xa_ln })
}

fn encoder<B: Backend>(wm: &mut WeightMap<B>, d: &ModelDimensions) -> Result<Encoder<B>> {
    let conv1 = Conv1d {
        weight: Param::from(wm.take_3d("encoder.conv1.weight", [d.n_audio_state, d.n_mels, 3])?),
        bias: Param::from(wm.take_1d("encoder.conv1.bias", d.n_audio_state)?),
    };
    let conv2 = Conv1d {
        weight: Param::from(wm.take_3d("encoder.conv2.weight", [d.n_audio_state, d.n_audio_state, 3])?),
        bias: Param::from(wm.take_1d("encoder.conv2.bias", d.n_audio_state)?),
    };
    let pos = Param::from(wm.take_2d("encoder.positional_embedding", [d.n_audio_ctx, d.n_audio_state])?);
    let blocks = (0..d.n_audio_layer)
        .map(|i| block(wm, &format!("encoder.blocks.{i}"), d.n_audio_state, d.n_audio_state * 4, d.n_head, false))
        .collect::<Result<Vec<_>>>()?;
    let ln_post = ln(wm, "encoder.ln_post", d.n_audio_state)?;
    Ok(Encoder { conv1, conv2, positional_embedding: pos, blocks, ln_post })
}

fn decoder<B: Backend>(wm: &mut WeightMap<B>, d: &ModelDimensions, n_text_ctx: usize) -> Result<Decoder<B>> {
    let tok = Param::from(wm.take_2d("decoder.token_embedding.weight", [d.n_vocab, d.n_text_state])?);
    let pos = Param::from(wm.take_2d("decoder.positional_embedding", [n_text_ctx, d.n_text_state])?);
    let blocks = (0..d.n_text_layer)
        .map(|i| block(wm, &format!("decoder.blocks.{i}"), d.n_text_state, d.n_text_state * 4, d.n_head, true))
        .collect::<Result<Vec<_>>>()?;
    let ln = ln(wm, "decoder.ln", d.n_text_state)?;
    Ok(Decoder { token_embedding: tok, positional_embedding: pos, blocks, ln })
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    pub dims: ModelDimensions,
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn from_weights(dims: ModelDimensions, wm: &mut WeightMap<B>, n_text_ctx: usize) -> Result<Self> {
        let encoder = encoder(wm, &dims)?;
        let decoder = decoder(wm, &dims, n_text_ctx)?;
        Ok(Self { dims, encoder, decoder })
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(&self, tokens: Tensor<B, 1, burn::tensor::Int>, xa: &Tensor<B, 3>) -> Tensor<B, 2> {
        self.decoder.forward(tokens, xa.clone())
    }
}
```

- [ ] **Step 4: pass** — `cargo test -p whisper-burn --test whisper`
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: assemble Whisper from WeightMap with strict weight drain"
```

---

### Task 17: Backend aliases (compile-time wgpu/ndarray)

**Files:**
- Create: `crates/whisper-burn/src/backends.rs`
- Create: `crates/whisper-burn/src/lib.rs` re-export
- Modify: `crates/whisper-burn/Cargo.toml` (features) — already done in Task 1; confirm here.

**Interface:**
- `pub type WgpuWhisper = Whisper<WgpuBackend>;` behind `#[cfg(feature = "wgpu")]`
- `pub type CpuWhisper = Whisper<NdarrayBackend>;` behind `#[cfg(feature = "ndarray")]`
- Expose convenience device constructors: `pub fn wgpu_device(adapter: WgpuAdapterSpec) -> Result<WgpuDevice>`, `pub fn cpu_device() -> NdarrayDevice`.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/backends.rs`:
```rust
// compile + type-level check; ndarray feature is always on in dev.
use whisper_burn::backends::{CpuWhisper, cpu_device};
#[test]
fn cpu_whisper_instantiates_with_tensor_device() {
    let _dev = cpu_device();
    let _t: burn::tensor::Tensor<burn::backend::ndarray::NdarrayBackend<f32>, 2> =
        burn::tensor::Tensor::from_data(burn::tensor::Data::from([[1.0f32]]), &_dev);
}
```
- [ ] **Step 2: fail** — `cargo test -p whisper-burn --features ndarray --test backends`
- [ ] **Step 3: implement** — `crates/whisper-burn/src/backends.rs`:
```rust
#[cfg(feature = "ndarray")]
pub type CpuWhisper = crate::model::whisper::Whisper<burn::backend::ndarray::NdarrayBackend<f32>>;
#[cfg(feature = "ndarray")]
pub fn cpu_device() -> burn::backend::ndarray::NdarrayDevice {
    burn::backend::ndarray::NdarrayDevice::default()
}
#[cfg(feature = "wgpu")]
pub type WgpuWhisper = crate::model::whisper::Whisper<burn::backend::wgpu::WgpuBackend>;
```
`lib.rs` re-export: `pub mod backends;`
- [ ] **Step 4: pass.** (wgpu compile can be checked separately: `cargo check -p whisper-burn --features wgpu`.)
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: backend feature aliases CpuWhisper/WgpuWhisper"
```

---

### Task 18: Model weight download (ureq)

**Files:**
- Create: `crates/whisper-burn/src/download.rs`
- Modify: `crates/whisper-burn/src/lib.rs`

**Interface:**
- `pub const HF_BASE: &str = "https://huggingface.co";`
- `pub fn repo_base(size: ModelSize) -> String` → `"{HF_BASE}/{size.repo_id()}/resolve/main"`
- `pub fn download_to(url: &str, dest: &Path, overwrite: bool) -> Result<PathBuf>` — ureq GET with retry ×3, writes atomically (`.part` then rename). No auth.
- `pub fn download_checkpoint(size: ModelSize, dest_dir: &Path, overwrite: bool, require_weights: bool) -> Result<PathBuf>` — fetches `config.json` (always) + `model.safetensors` (if `require_weights`), where `dest_dir` is `~/.cache/whisper-burn/{repo_id}`.
- Cache layout: `dest/model.safetensors`, `dest/config.json`.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/download.rs` (guarded `#![cfg(feature = "weights")]`) uses `#[ignore]` (network):
```rust
#![cfg(feature = "weights")]
use whisper_burn::config::ModelSize;
use whisper_burn::download::download_checkpoint;

#[test]
#[ignore]
fn downloads_tiny() {
    let dir = std::env::temp_dir().join("wburn_test_tiny");
    std::fs::create_dir_all(&dir).unwrap();
    let _p = download_checkpoint(ModelSize::Tiny, &dir, true, true).unwrap();
    assert!(dir.join("model.safetensors").exists());
    assert!(dir.join("config.json").exists());
}
```
- [ ] **Step 2: fail** — `cargo test -p whisper-burn --features weights --test download -- --ignored` (may fail offline; acceptable to skip with `--include-ignored` when network available). Expected initial: FAIL (missing module).
- [ ] **Step 3: implement** — `download.rs`. Note: run `cargo add ureq@3` etc. Handle `ureq::Error::Status`, retry on `Transport` errors in a loop.
- [ ] **Step 4: pass** (network-dependent; `#[ignore]` keeps CI offline-friendly).
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: download checkpoints from Hugging Face via ureq"
```

---

### Task 19: `Whisper::load` / `from_pretrained`

**Files:**
- Modify: `crates/whisper-burn/src/model/whisper.rs`
- Create: `crates/whisper-burn/src/weights/mod.rs` (re-export loader/safetensors)
- Create helper test generator: `crates/whisper-burn/tests/common/mod.rs` extracting the tiny fake-checkpoint builder sketched in Task 16.

**Interface:**
- `impl<B: Backend> Whisper<B> {`
  - `pub fn load(size: ModelSize, weights_dir: &Path, device: B::Device) -> Result<Self>` — reads `weights_dir/config.json` (built from `crate::config::read_config`), validates vs `ModelSize` expectations, reads `weights_dir/model.safetensors`, builds `WeightMap::new`, calls `Self::from_weights`, then `wm.finish()` **must** pass. Errors: `Error::MissingWeights` if `model.safetensors` absent, `Error::MissingWeight` per missing key, `Error::UnexpectedWeight` on leftovers.
  - `pub fn from_pretrained(size: ModelSize, device: B::Device) -> Result<Self>` — `weights_dir = ~/.cache/whisper-burn/{repo_id}` (respect `WHISPER_BURN_CACHE` env override); if `model.safetensors` missing and `feature = "download"`/`"weights"` is on, `download_checkpoint`; else `Error::MissingWeights` with a message telling the user to run with the `weights` feature. Then delegates to `load`.
  - `pub fn config_path(size)` helper.
- Note: `from_pretrained` needs no `audio` feature; decoding feature (`audio`) is independent.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/load.rs`:

```rust
use burn::backend::ndarray::{NdarrayBackend, NdarrayDevice};
use whisper_burn::config::ModelSize;
use whisper_burn::model::whisper::Whisper;
use whisper_burn::tests::common::write_fake_checkpoint; // from tests/common

type B = NdarrayBackend<f32>;

#[test]
fn load_from_fake_checkpoint_succeeds_and_drains() {
    let tmp = std::env::temp_dir().join("wburn_load_test");
    std::fs::create_dir_all(&tmp).unwrap();
    write_fake_checkpoint(ModelSize::Tiny, &tmp);  // config.json + model.safetensors (all zeros)
    let w = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdarrayDevice::default()).unwrap();
    assert_eq!(w.dims.n_vocab, ModelSize::Tiny.expected_dims().n_vocab);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn load_missing_weights_is_typed_error() {
    let tmp = std::env::temp_dir().join("wburn_load_missing");
    std::fs::create_dir_all(&tmp).unwrap();
    let e = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdarrayDevice::default()).unwrap_err();
    assert!(matches!(e, whisper_burn::Error::MissingWeights { .. }), "got {e:?}");
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn missing_single_weight_reports_key() {
    // write a checkpoint then delete `encoder.ln_post.bias` from the safetensors → MissingWeight
    let tmp = std::env::temp_dir().join("wburn_load_rm");
    std::fs::create_dir_all(&tmp).unwrap();
    write_fake_checkpoint(ModelSize::Tiny, &tmp);
    // (impl: helper `remove_key(path, key)` that rebuilds safetensors without that tensor)
    // whisper_burn::tests::common::remove_key(&tmp.join("model.safetensors"), "encoder.ln_post.bias");
    let e = Whisper::<B>::load(ModelSize::Tiny, &tmp, NdarrayDevice::default()).unwrap_err();
    assert!(matches!(e, whisper_burn::Error::MissingWeight(_)), "got {e:?}");
    std::fs::remove_dir_all(&tmp).unwrap();
}
```

- [ ] **Step 2: fail** — `cargo test -p whisper-burn --test load` (needs `common` module wired as `#[path]`; add `tests/common` to `Cargo.toml` as `[[test]]` harness? Use `#[path = "common/mod.rs"] mod common;` inside `load.rs`).
- [ ] **Step 3: implement**

Add to `crates/whisper-burn/src/model/whisper.rs` (plus imports): `use crate::config::read_config; use crate::weights::safetensors::read_safetensors; use crate::weights::loader::WeightMap;`

```rust
pub fn load(size: ModelSize, weights_dir: &Path, device: B::Device) -> Result<Self> {
    let cfg_path = weights_dir.join("config.json");
    let st_path = weights_dir.join("model.safetensors");
    if !st_path.exists() {
        return Err(Error::MissingWeights { path: st_path });
    }
    let dims = read_config(&cfg_path, size)?;
    let bytes = std::fs::read(&st_path).map_err(|e| Error::Io { path: st_path.clone(), source: e })?;
    let st = read_safetensors(&bytes)?;
    let mut wm = WeightMap::new(st, bytes, device);
    let model = Self::from_weights(dims, &mut wm, dims.n_text_ctx)?;
    wm.finish()?;
    Ok(model)
}

#[cfg(any(feature = "weights", feature = "download"))]
pub fn from_pretrained(size: ModelSize, device: B::Device) -> Result<Self> {
    let dir = crate::download::cache_dir(size)?;
    if !dir.join("model.safetensors").exists() {
        let _ = crate::download::download_checkpoint(size, &dir, true, true)?;
    }
    Self::load(size, &dir, device)
}

#[cfg(not(any(feature = "weights", feature = "download")))]
pub fn from_pretrained(_size: ModelSize, _device: B::Device) -> Result<Self> {
    Err(Error::MissingWeights {
        path: std::path::PathBuf::from("(weights feature disabled; enable download or weights)"),
    })
}
```
> `read_config` in Task 3 returns `ModelDimensions` validated against `ModelSize`; ensure it reads `max_source_positions`/`max_target_positions` for `n_audio_ctx`/`n_text_ctx`. `Generic.Added` instruction `n_text_ctx` = config `max_target_positions` (tiny 448); `n_audio_ctx` = `max_source_positions` (1500).

Include: `crate::weights::safetensors` + `loader` re-export under `mod` in `lib.rs`. `download::cache_dir(size)`:
```rust
pub fn cache_dir(size: ModelSize) -> Result<PathBuf> {
    Ok(if let Ok(p) = std::env::var("WHISPER_BURN_CACHE") {
        PathBuf::from(p).join(size.repo_id())
    } else {
        home_dir()?.join(".cache").join("whisper-burn").join(size.repo_id())
    })
}
```
(`home_dir()` = `std::env::var_os("HOME")` / `USERPROFILE` on Windows.)

- [ ] **Step 4: pass** — update `tests/common/mod.rs` `write_fake_checkpoint` to include the full tiny weight set (reuse Task 16 generator; zeros work for `load` since we only test shape/drain).
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: load Whisper from local checkpoint and from_pretrained via cache"
```

---

### Task 20: `forward_mel` public + first encoder pass on real checkpoint

**Rationale:** two hard parts remain (deterministic decode) but before any of them we must verify the *encoder* against the reference on a real tiny checkpoint — this catches conv/layout/mel mismatches early, before investing in beam logic.

**Files:**
- Create: `crates/whisper-burn/tests/golden_encoder.rs` (ignored)

**Interface:** (in `model/mod.rs`) `pub struct WhisperConcrete` re-exports, plus `whisper_burn::features::mel::FeatureExtractor` integration:
`pub fn extract_features(audio_pcm: &[f32]) -> Vec<f32>` (mel `[n_mels, frames]`) — from Task 8. Re-export for the CLI/golden harness.

- [ ] **Step 1: failing test** — `golden_encoder.rs` (requires network download of tiny weights):

```rust
#![cfg(all(feature = "ndarray", feature = "audio"))]
use burn::backend::ndarray::NdarrayDevice;
use whisper_burn::config::ModelSize;
use whisper_burn::model::whisper::Whisper;
use whisper_burn::features::window::split_chunks;
use whisper_burn::features::mel::FeatureExtractor;

#[test]
#[ignore = "network + golden files"]
fn golden_encoder_matches_reference() {
    let dev = NdarrayDevice::default();
    let w = Whisper::<burn::backend::ndarray::NdarrayBackend<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    // 1.0 s of a 440 Hz tone sustained, 16 kHz
    let pcm: Vec<f32> = (0..16000)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let chunks = split_chunks(&pcm);
    let fe = FeatureExtractor::new(w.dims.n_mels, 16000).unwrap();
    let mel = fe.log_mel(&chunks[0].0).unwrap(); // [n_mels, 3000]
    let mel = burn::tensor::Tensor::<burn::backend::ndarray::NdarrayBackend<f32>, 3>
        ::from_data(burn::tensor::Data::new(mel, [1, w.dims.n_mels, 3000]), &dev);
    let enc = w.forward_encoder(mel);
    let _ = enc; // assert values after cementing in Task 27
    // TODO: compare head against `python whisper` reference segment — cemented in Task 27.
}
```
- [ ] **Step 2: fail** — `cargo test -p whisper-burn --features "ndarray audio" --test golden_encoder -- --ignored`
- [ ] **Step 3: implement** — wire FeatureExtractor + forward_encoder (already exist; this task is largely harness plumbing). Key: confirm `FeatureExtractor::new(n_mels, sr)` uses our baked filterbanks; if encoder mismatch appears, debug systematically — check conv layout (row-major `[out, in, k]`) and mel filter orientation first.
- [ ] **Step 4: pass** — run with network; compare against reference log-mel/encoder numerically with tolerance (cement numbers in Task 27 fixture files).
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "test: golden encoder pass against tiny reference"
```

---

### Task 21: Language detection + no-speech classifier

**Files:**
- Create: `crates/whisper-burn/src/decoding/lang.rs`

**Interface:**
- `pub struct LanguageInfo { pub token: u32, pub language: &'static str, pub prob: f32 }`
- `pub fn detect_language<B: Backend>(whisper: &Whisper<B>, xa: &Tensor<B,3>, options...) -> Result<LanguageInfo>`
  - Decoder init `sot_sequence = [sot, tokenizer.lang_token(lang)]` (language `None` for auto).
  - Run greedy one step, read `logits[-1]` restricted to the language-token window `[50258, 50258 + n_languages]` (± `0.0`), pick argmax, probability `p = softmax(logits_window)[arg]`.
- Deterministic, matches whisper reference ordering: language ranking is by the decoder logits.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/lang.rs` (guarded `--ignored` / net):
```rust
#![cfg(feature = "ndarray")]
use burn::backend::ndarray::NdarrayDevice;
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::lang::detect_language;
use whisper_burn::model::whisper::Whisper;
type B = burn::backend::ndarray::NdarrayBackend<f32>;

#[test]
#[ignore = "network"]
fn detects_english_silence() {
    let dev = NdarrayDevice::default();
    let w = Whisper::<B>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    // silence → engines usually pick english
    let pcm = vec![0.0f32; 480_000];
    let f = whisper_burn::features::mel::FeatureExtractor::new(80, 16000).unwrap();
    let chunks = whisper_burn::features::window::split_chunks(&pcm);
    let mel = f.log_mel(&chunks[0].0).unwrap();
    let xa = w.forward_encoder(Tensor::from_data(Data::new(mel, [1, 80, 3000]), &dev));
    let info = detect_language(&w, &xa).unwrap();
    assert_eq!(info.language, "en");
}
```
- [ ] **Step 2: fail**
- [ ] **Step 3: implement** — `decoding/lang.rs`; uses `whisper.tokenizer.lang_token(code)` and `decoder.forward`; needs `n_audio_ctx` mask drop. Note the reference restricts to `tokenizer.language_token` range with a `-inf` mask on the other slots (see `_get_initial_prompt/_temperature`?), simpler: build 1-step logits then slice `[.., lang0..lang0+n_langs]`, argmax, softmax prob. Keep `LanguageInfo.prob` = softmax over the window.
- [ ] **Step 4: pass** (network net-ignored)
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: decoder language detection from 1-step sot logits"
```

---

### Task 22: Greedy decoding (temperature 0) + no-speech

**Files:**
- Create: `crates/whisper-burn/src/decoding/mod.rs`
- Create: `crates/whisper-burn/src/decoding/greedy.rs`

**Interface:**
- `pub struct GreedyOptions { pub max_tokens: usize, pub sample_len: usize, pub suppress_blank: bool, pub suppress_tokens: Vec<u32>, pub no_speech_threshold: f64, pub logprob_threshold: f64, pub condition_on_previous_text: bool }` (defaults = whisper CLI: `suppress_blank=True`, `suppress_tokens="-1"`→ all special tokens, `no_speech_threshold=0.6`, `logprob_threshold=-1.0`).
- `pub fn greedy_search<B: Backend>(whisper: &Whisper<B>, xa: &Tensor<B,3>, tokens: &mut Vec<u32>, options: &GreedyOptions) -> Result<()>`
  - Loop: `logits = decoder(tokens)`, apply suppression: zero-out `suppressed_ids` (incl. all `>= n_vocab`, everything in `tokenizer.non_speech_tokens`, and id `eot`+1 (whisper suppresses `eot+1`? no, whisper suppresses `blank=-1`→sot) — keep faithful list), `logits[.., -1]`, argmax; if `== eot` → stop (record `mean_logprob`), else push.
  - On hitting `max_tokens`, stop.
- no-speech check lives in transcribe (Task 24); greedy is purely token loop.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/greedy.rs` (net-ignored):
```rust
#![cfg(feature = "ndarray")]
use burn::backend::ndarray::NdarrayDevice;
use burn::tensor::{Data, Tensor};
use whisper_burn::config::ModelSize;
use whisper_burn::decoding::greedy::{greedy_search, GreedyOptions};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::tokenizer::whisper::TextTokenizer;
type B = burn::backend::ndarray::NdarrayBackend<f32>;

#[test]
#[ignore = "network"]
fn greedy_silence_is_blank_predictable() {
    let dev = NdarrayDevice::default();
    let w = Whisper::<B>::from_pretrained(ModelSize::Tiny, dev).unwrap();
    let fe = whisper_burn::features::mel::FeatureExtractor::new(w.dims.n_mels, 16000).unwrap();
    let pcm = vec![0.0f32; 480_000];
    let chunks = whisper_burn::features::window::split_chunks(&pcm);
    let mel = fe.log_mel(&chunks[0].0).unwrap();
    let xa = w.forward_encoder(Tensor::from_data(Data::new(mel, [1, w.dims.n_mels, 3000]), &dev));
    let tok = TextTokenizer::new(w.dims.n_vocab);
    let mut tokens = tok.sot_sequence(None);
    let opts = GreedyOptions::default();
    greedy_search(&w, &xa, &mut tokens, &opts).unwrap();
    // silence + tiny → usually the full sot … eot quickly; just assert type-safety + terminates
    assert!(!tokens.is_empty());
    assert!(tokens.len() < 448);
}
```
- [ ] **Step 2: fail** (missing module)
- [ ] **Step 3: implement** — `decoding/greedy.rs`; suppression helper `fn apply_suppression(logits: Vec<f32>, tokens: &Vec<u32>)`; mark `mean_logprob` exported for Task 24.
- [ ] **Step 4: pass** (network)
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: greedy temperature-0 search over decoder tokens"
```

---

### Task 23: Beam search with timestamps (deterministic path)

**Files:**
- Create: `crates/whisper-burn/src/decoding/beam.rs`

**Interface:**
- `pub struct BeamOptions { pub beam_size: usize, pub best_of: usize, pub patience: f32, pub max_tokens: usize, pub suppress_blank: bool, pub suppress_tokens: Vec<u32>, pub condition_on_previous_text: bool }`
- `pub fn beam_search<B: Backend>(whisper, xa, tokens: &mut Vec<u32>, options) -> Result<()>`
  - Standard whisper algorithm: keep `beam_size` hypotheses `(tokens, sum_logprob, avg_logprob)`, expand by top-`beam_size` argmax per beam each step (deterministic, no random sampling at temperature 0), apply suppression masks per beam, track `last_avg_prob`. Stop per-beam at `eot` (`finished` list) until all finished or `max_tokens`.
  - Must be bit-reproducible vs reference *greedy* on a short clip (patience 0, beam 1) and vs reference beam (patience 1) on a real clip — cemented in Task 27.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/beam.rs` (net-ignored; mirrors greedy test shape, asserts `len<448`, termination, and best `tokens` starts with `sot_sequence`).
- [ ] **Step 2: fail**
- [ ] **Step 3: implement** — `decoding/beam.rs`
- [ ] **Step 4: pass** (network); add a second assertion: `beam 1 == greedy` output on the same prompt (verifies both paths agree deterministically).
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: beam search decoding keeping top-k hypotheses"
```

---

### Task 24: Timestamps + segment assembly (reference format)

**Files:**
- Create: `crates/whisper-burn/src/decoding/segments.rs`

**Interface:**
- `pub struct Segment { pub start: u32, pub end: u32, pub text: String }` — times in ms.
- `pub fn timestamp_token_to_ms(token: u32, tokenizer: &TextTokenizer) -> u32` → `(token - timestamp_begin) * TIME_PRECISION_MS` with `TIME_PRECISION_MS = 10.0` (whisper `time_precision = 1.0 / sample_rate * hop_length * 1000` = 10 ms).
- `pub fn assemble_segments(tokens: &[u32], tokenizer: &TextTokenizer, seek: u32, sample_rate: u32) -> (Vec<Segment>, f32)` — parse `[sot_sequence][timestamps|text]*[eot]`, split text runs between timestamp tokens into `Segment { start, end, text }` with text = detokenized bytes, filtering empty. Whisper adds `"(timestamp_begin < text_nothing"`... keep the reference behavior: a segment's `start` = first timestamp, `end` = second or `+length`; text joined with spaces; skip `<|0.00|>` placeholder first timestamp. Return `(segments, avg_logprob)`.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/segments.rs` (unit, no net):
```rust
use whisper_burn::decoding::segments::{timestamp_token_to_ms, assemble_segments};
use whisper_burn::tokenizer::whisper::TextTokenizer;

#[test]
fn timestamp_ms_from_token() {
    let tok = TextTokenizer::new(51865);
    let t0 = tok.timestamp_token(0);    // 50364
    let t1 = tok.timestamp_token(1);    // 50365
    assert_eq!(timestamp_token_to_ms(t0, &tok), 0);
    assert_eq!(timestamp_token_to_ms(t1, &tok), 10);
}

#[test]
fn segments_split_on_timestamps() {
    let tok = TextTokenizer::new(51865);
    let sequence = vec![
        50364,                       // |0.00|
        tok.encode("hello").unwrap()[0], // "hello"  (single-byte token in tiny vocab; use encode_ordinary)
        50369,                       // |0.05|
        tok.encode("world").unwrap()[0],
        50372,                       // |0.08|
        tok.eot,
    ];
    let (segs, _) = assemble_segments(&sequence, &tok, 0, 16000);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 0);   // first timestamp → start
    assert_eq!(segs[0].end, 50);    // second timestamp marks end
    assert_eq!(segs[0].text, "hello");
    assert_eq!(segs[1].start, 50);
    assert_eq!(segs[1].end, 80);
    assert_eq!(segs[1].text, "world");
}
```
- [ ] **Step 2: fail**
- [ ] **Step 3: implement** — `segments.rs`
- [ ] **Step 4: pass**
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: segment assembly from timestamp-delimited token streams"
```

---

### Task 25: `Whisper::transcribe` orchestration (full pipeline)

**Files:**
- Modify: `crates/whisper-burn/src/model/whisper.rs`
- Create: `crates/whisper-burn/src/transcribe.rs`

**Interface:**
- `pub struct TranscriptionOptions { pub language: Option<Language>, pub task: Task /* Transcribe|Translate */, pub beam_size: usize, pub best_of: usize, pub temperature: f32, pub condition_on_previous_text: bool, pub no_speech_threshold: f32, pub logprob_threshold: f32, pub initial_prompt: Option<String> }
  -- v1: temperature must be `0.0` (returns `Err(Unsupported) if != 0.0); beam_size 0→greedy, ≥1→beam.
- `pub struct TranscriptionSegment { pub start: u32, pub end: u32, pub text: String }`
- `pub fn transcribe<B: Backend>(whisper: &Whisper<B>, pcm: &[f32], sample_rate: u32, options: &TranscriptionOptions) -> Result<Vec<TranscriptionSegment>>` — top-level: resample (if !=16k), split into 30s chunks, per chunk: mel → encoder → language detect (if None) → greedy/beam → no-speech check (avg_logprob < logprob_threshold && max_text?) → assemble segments offset by `seek`; concat, filter `condition_on_previous_text` prefix = last 'non-start' tokens.

- [ ] **Step 1: failing test** — `crates/whisper-burn/tests/transcribe.rs` (net-ignored): run on a 1 s real WAV with `language: Some("en")`, beam 0; assert `segments.len() >= 1`, `segments[0].start == 0`, text non-empty. Second: translate task on same clip returns text (not timestamped `en`).
- [ ] **Step 2: fail**
- [ ] **Step 3: implement** — orchestration in `transcribe.rs`. Wire mel/chunk/encoder (Task 20/21), lang detect, decode (greedy/beam), no-speech, `assemble_segments` offset by chunk seek, prev-text conditioning (append previous tokens not `eot/start`).
- [ ] **Step 4: pass** (network; only do this once both 21–24 pass on the clip)
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn
git commit -m "feat: end-to-end transcribe pipeline with 30s chunking"
```

---

### Task 26: CLI (`whisper-burn` binary)

**Files:**
- Create: `crates/whisper-burn-cli/src/main.rs`
- Modify: `crates/whisper-burn-cli/Cargo.toml` (add clap dependecy; wire `whisper-burn` with `features = ["audio", "download", default]`)

**Interface:**
- `whisper-burn [OPTIONS] <audio>` — clap derive.
- Options: `--model <tiny|base|small|medium|large-v2|large-v3|...>` default `tiny`; `--language <code>` optional; `--task <transcribe|translate>` default transcribe; `--beam-size <N>` default 0 (greedy); `--language`, `--verbose`, `--output <txt|srt|vtt|json>` default `txt`; `--device <wgpu|cpu>` default `wgpu`.
- Prints segments `[00:00.000 --> 00:00.500]  text` to stdout (and `.txt`/`.srt`/`.vtt`/`.json` files next to audio when `--output` set).

- [ ] **Step 1: failing test** — `crates/whisper-burn-cli/tests/cli.rs` (integration; ignored for net):
```rust
#![cfg(feature = "audio")]
#[test]
#[ignore = "network"]
fn cli_prints_a_transcription() {
    // run `cargo run -p whisper-burn-cli -- <tiny-wav>` via std::process::Command
    // assert stdout contains a timestamp arrow and non-empty text
}
```
- [ ] **Step 2: fail** (no main) — `cargo build -p whisper-burn-cli`
- [ ] **Step 3: implement** — `main.rs`: parse args → choose backend device (wgpu default flame-adapter arg; cpu via `--device cpu`) → `Whisper::from_pretrained` → decode audio → `transcribe` → print segments. Use `whisper-burn::audio::decode_to_mono_f32` + `resample_to_16k`.
- [ ] **Step 4: pass** (compile + run on a short clip; network ignored in CI)
- [ ] **Step 5: commit**
```bash
git add crates/whisper-burn-cli
git commit -m "feat: whisper-burn CLI reading audio and printing transcripts"
```

---

### Task 27: Acceptance — cement goldens vs reference whisper

**Files:**
- Create: `crates/whisper-burn/tests/golden.rs` (ignored), `tests/fixtures/reference_tiny/` (log-mel npy, encoder npy, greedy/logits npy, `expected_segments.txt`)
- Create: `scripts/cement_reference.py` (Python, "cements" reference outputs into `/tests/fixtures` from the official `openai/whisper` pipeline)

**Steps (manual, gated by `#[ignore]`):**
1. Run `scripts/cement_reference.py` once with `python` + `openai-whisper` + `--model tiny` to produce golden files for a canonical clip (a known short WAV, e.g. the classic `jfk.flac` 11 s excerpt resampled to 16k mono). Store:
   - `log_mel.npy` (float32 `[n_mels, 3000]`), `encoder.npy` `[1, 1500, n_state]`, `segments.txt`, plus the *raw greedy token ids* via `decode_token_level` (npy int32 `[n]`).
2. Rust `golden.rs` tests load the npy fixtures, run our `FeatureExtractor::log_mel`, `forward_encoder`, greedy/beam, and assert **allclose** (`atol=1e-4` mel, `atol=1e-3` encoder) and exact `segments.txt` text equality.
3. If drift: fix in the smallest layer (mel filters orientation → encoder conv → decoder greed), re-cement, re-run. Do NOT loosen tolerances without documenting why.
4. Final gate: `cargo test -p whisper-burn --features "ndarray audio download" --test golden -- --ignored` all green, plus `cargo test -p whisper-burn` (offline suite) green, plus `cargo clippy --all-targets --all-features` clean.

- [ ] Evaluate log-mel vs reference (fix mel.rs if bins/drop-last mismatch found)
- [ ] Evaluate encoder vs reference (fix conv/order)
- [ ] Evaluate greedy tokens vs reference (fix decoder/attention/suppression)
- [ ] Evaluate beam vs reference (`--beam-size 4 --patience 1` on `jfk.flac`)
- [ ] Evaluate transcribe text vs `segments.txt` (fix chunk/prefix/hang)
- [ ] Commit: `git add . && git commit -m "test: cement golden reference-transcription fixtures"`

---