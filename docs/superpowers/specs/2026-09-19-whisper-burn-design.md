# Whisper Inference in Rust + Burn — Design

- **Date:** 2026-09-19
- **Status:** Draft for review (brainstorming; sections 1–3 approved; review feedback folded in 2026-09-19)
- **Path classification:** Architectural (new multi-crate project)

## 1. Goal

Build a faithful, from-scratch Rust implementation of OpenAI Whisper **inference**
using the [Burn](https://github.com/tracel-ai/burn) deep-learning framework, that:

- loads the official `openai/whisper-*` pre-trained checkpoints (safetensors),
- transcribes real audio into text,
- reproduces OpenAI's reference behavior (audio frontend, model, tokenizer,
  full decoding: greedy + beam search, timestamps, language detection,
  no-speech detection, temperature fallback, transcribe/translate).

Success = the same transcription (token-for-token under deterministic settings)
as the `openai/whisper` Python reference for the supported models.

## 2. Deliverables

A Cargo **workspace** producing two artifacts:

1. `whisper-burn` — **library crate** with a clean public API, reusable from other
   apps (e.g. a Tauri/web or desktop app). PCM-in, structured-text-out.
2. `whisper-burn-cli` — thin **binary/CLI** wrapping the library, adding audio-file
   decoding and a `download` command.

## 3. Decisions (from brainstorming Q&A)

| Topic | Decision |
|---|---|
| End goal | Full Whisper transcription with real pre-trained weights |
| Model family | Multilingual; support tiny/base/small/medium/large/large-v2/large-v3/large-v3-turbo |
| Backend | Model generic over Burn `Backend`; **wgpu (default) / ndarray chosen at compile time via features** |
| Weights | Download `openai/whisper-*` **safetensors** from HuggingFace; **pure-Rust** load (no Python) |
| Audio input | **Any common audio file**; decode via `symphonia`, resample to 16 kHz mono |
| Decoding | **Greedy + beam + timestamps + language auto-detect + no-speech** in v1; stochastic temperature fallback deferred to v2 |
| Implementation style | **Hybrid (Option C)**: implement the interesting parts by hand (safetensors reader, tiktoken BPE tokenizer, mel frontend, model, decoding); use crates only for boring IO/foundations |
| Crate layout | Workspace: reusable lib crate + thin bin crate |

Non-goals for v1: training/fine-tuning; stochastic temperature fallback; torch
`.bin` pickle loading (stretch); streaming/real-time; non-multilingual (`.en`)
checkpoints.

## 4. Architecture

### 4.1 Workspace layout

```
whisper_burn/                        # workspace root (Cargo.toml [workspace])
  docs/superpowers/specs/            # this spec
  crates/
    whisper-burn/                    # LIB crate: the public API (reusable)
      src/
        lib.rs          # public API surface (below)
        error.rs        # Error enum (thiserror)
        backend.rs      # backend re-exports + WgpuWhisper/CpuWhisper aliases + default_device()
        config.rs       # ModelDimensions (parsed from config.json)
        weights/
          safetensors.rs
          loader.rs
          download.rs
        tokenizer/
          tiktoken.rs
          whisper.rs
        features/
          stft.rs
          mel.rs
          window.rs
        audio/
          decode.rs      # symphonia (behind optional `audio` feature)
          resample.rs    # rubato / fallback
        model/
          attention.rs
          block.rs
          encoder.rs
          decoder.rs
          whisper.rs
        decoding/
          strategy.rs
          language.rs
          timestamps.rs
          transcribe.rs
    whisper-burn-cli/                # BIN crate: thin CLI wrapper
      src/main.rs     # clap: `download <size>`, `transcribe <file> [options]`
  tests/              # workspace-level integration tests + fixtures
```

### 4.2 Public API (lib crate)

Audio-format-agnostic: the lib takes already-decoded PCM + sample rate, so a
frontend (Tauri JS, etc.) can feed samples directly. File decoding lives in the
optional `audio` feature / the CLI.

```rust
pub type WgpuWhisper = Whisper<Wgpu>;    // enabled by feature "wgpu" (default)
pub type CpuWhisper  = Whisper<NdArray>; // enabled by feature "ndarray"

pub enum ModelSize { Tiny, Base, Small, Medium, Large, LargeV2, LargeV3, LargeV3Turbo }

pub struct Whisper<B: Backend> { /* WhisperModel<B> + tokenizer + feature extractor + device */ }

impl<B: Backend> Whisper<B> {
    /// Load from a dir already containing config.json + model.safetensors + multilingual.tiktoken.
    pub fn load(size: ModelSize, weights_dir: impl AsRef<Path>, device: &B::Device) -> Result<Self>;

    /// Download if missing, then load. One-stop convenience over `download` + `load`.
    pub fn from_pretrained(size: ModelSize, cache_dir: impl AsRef<Path>, device: &B::Device) -> Result<Self>;

    /// Download only; cache-checked. Returns the weights dir.
    pub fn download(size: ModelSize, into_dir: impl AsRef<Path>) -> Result<PathBuf>;

    pub fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        opts: &TranscribeOptions,
    ) -> Result<Transcription>;
}

pub struct TranscribeOptions {
    pub language: Option<Language>, // None = auto-detect
    pub task: Task,                 // Transcribe | Translate
    pub timestamps: bool,
    pub beam_size: Option<usize>,   // None = greedy; Some(n) = beam search
}

pub struct Transcription { pub text: String, pub segments: Vec<Segment>, pub language: String }
pub struct Segment { pub id: u32, pub start: f64, pub end: f64, pub text: String }
```

**Backend strategy: compile-time selection, not a runtime enum.** Burn's `Backend`
trait is large; a hand-written runtime enum would require delegating every tensor
op and would prevent the model from staying generic. All model + decoding code is
generic over `B: Backend`; the concrete backend is chosen at **build time via
Cargo features** (`wgpu`, default, or `ndarray`). The public API hides `B` behind
the `WgpuWhisper` / `CpuWhisper` aliases above.

- `Whisper<B>` owns the `B::Device` (set in `load`/`from_pretrained`) so callers
  don't thread devices through `transcribe`.
- `load` errors with `Error::MissingWeights` if files are absent; `from_pretrained`
  is the one-stop path.
- Aim for `Whisper<B>: Send + Sync` so frontends can run it in a worker thread;
  document any backend type that prevents this.
- The CLI selects the backend via the same feature flags; it prints a clear hint
  if built without a GPU-capable backend.

## 5. Model sizes & dimensions

Whisper variants are architecturally identical except for these knobs, so the
encoder/decoder/decoding code has **no per-size branches**:

| Model | n_mels | enc layers | dec layers | state | heads | n_vocab |
|---|---|---|---|---|---|---|
| tiny / base / small | 80 | 4 / 6 / 12 | 4 / 6 / 12 | 384 / 512 / 768 | 6 / 8 / 12 | 51865 |
| medium | 80 | 24 | 24 | 1024 | 16 | 51865 |
| large / large-v2 | 80 | 32 | 32 | 1280 | 20 | 51865 |
| large-v3 | **128** | 32 | 32 | 1280 | 20 | **51866** |
| large-v3-turbo | **128** | 32 | **4** | 1280 | 20 | 51866 |

Two gotchas absorbed by the design:

1. **large-v3 switched the mel frontend from 80 → 128 bins** (encoder conv1
   in-channels follows it).
2. **large-v3 added the `yue` language token appended at id 51865** to keep older
   ids stable — hence `n_vocab = 51866`.

**Dimensions are parsed from the checkpoint's `config.json` (authoritative), not
hardcoded.** `ModelSize` only maps to the HF repo name and validates the parsed
dims against the expected family. `features/mel.rs` is parameterized by `n_mels`;
conv1 in-channels come from config; `tokenizer/whisper.rs` builds its
special-token/language table from `n_vocab` + the language set.

Performance note: large/large-v3 on the ndarray CPU backend is supported but very
slow; wgpu is the practical path. Turbo (4 decoder layers) is much lighter.

## 6. Data flow

```
audio file → [symphonia decode] → PCM f32 interleaved
           → [resample] → 16 kHz mono f32
lib: transcribe():
   → [window] split into 30s chunks (pad last), 3000 frames / 480k samples each
   → [log-mel] STFT(Hann 400, hop 160) → power → ×mel filters(n_mels) → log10,
      clamp 1e-10, max(x, max−8), (x+4)/4                       → [n_mels × 3000]
   → [encoder] conv1(n_mels→d, k3, pad1) + GELU + conv2(d→d, k3, s2, pad1) + GELU
      → + positional_embedding (1500 rows, used in full)        → 1500 encoder tokens
   → [decoder] causal self-attn (448 ctx) + cross-attn (1500 kv); tied embedding head
   → [decode loop] per chunk: lang-detect / no-speech + greedy|beam w/ temperature
      fallback, optional timestamps → segments
   → [assembly] concatenate chunk text/segments → Transcription
```

Frame math (matches `openai/whisper/model.py` exactly): conv1 `k3, pad1` keeps
3000 frames → 3000; conv2 `k3, s2, pad1` → `(3000+2−3)/2+1 = 1500` = `n_audio_ctx`;
decoder cross-attends to 1500 encoder tokens; `positional_embedding` (1500 rows)
used in full (slice `[:1500]`).

## 7. Fidelity-critical numerics

Must match `whisper/audio.py`, `model.py`, `tokenizer.py` or pre-trained weights
produce garbage. Each item gets golden-value tests against the reference.

1. **Log-mel** — `n_mels` = 80 (tiny…large-v2, medium) or 128 (large-v3*),
   `sr=16000`, `n_fft=400`, `hop=160`, `fmin=0`, `fmax=8000`;
   `clamp(1e-10).log10()`; `max(x, max−8)`; `(x+4)/4`.
   **Filterbank: bake the official coefficients.** Whisper's librosa/slaney
   triangular filters are extracted once from the official `mel_filters.npz` and
   committed as static assets (`assets/mel_filters_80.bin` / `_128.bin`, f32
   row-major), rather than re-derived in Rust, to avoid float drift. The log-mel
   frontend runs on the **host in f32** (not a Burn op) — only the model runs on
   the device.
2. **STFT** — torch `stft` semantics with `center=True` (constant-zero pad of
   `n_fft/2` on both sides), Hann window, magnitude² over the first `n_fft/2+1`
   bins. DFT-first implementation (O(frames·400)); `rustfft` is a documented
   optimization.
3. **Tokenizer** — byte-level BPE over Whisper's `multilingual.tiktoken`, with the
   reference `pat_str` regex split before merges. Special token ids from
   `whisper/tokenizer.py` (eot=50256, sot=50257, translate/transcribe/startoflm/
   startofprev/nospeech/notimestamps, per-language tokens, timestamp ids), with the
   language table + `n_vocab` handling `yue`/large-v3. Constants pinned by
   round-trip + golden tests. **Exact ids to be transcribed verbatim from the
   reference during implementation.**
4. **Activations/ops** — erf-based GELU (torch default), LayerNorm eps=1e-5,
   attention `q·kᵀ/√d_head` → softmax. Standard Burn ops; golden-test each.

## 8. Weight loading, config, backend

### 8.1 Download (`weights/download.rs`, `ureq`)

`ModelSize` → HF repo `openai/whisper-{tiny,base,small,medium,large,large-v2,large-v3,large-v3-turbo}`.
Fetch `config.json`, `model.safetensors`, `multilingual.tiktoken` into
`dir/{size}/`. Cache-checked so re-runs are cheap. Safetensors is the primary
format; missing → clear error. Torch `.bin` pickle fallback is out of scope for v1.
**First implementation task verifies each target repo actually publishes
`model.safetensors`.**

### 8.2 Config (`config.rs`)

Parse HF `config.json` → `ModelDimensions`. **Explicit HF key mapping** (HF names
differ from Whisper's internal names):

| internal | HF `config.json` key |
|---|---|
| `n_mels` | `num_mel_bins` |
| `n_audio_layer` | `encoder_layers` |
| `n_text_layer` | `decoder_layers` |
| `n_audio_state` / `n_text_state` | `d_model` |
| `n_head` | `encoder_attention_heads` |
| `n_vocab` | `vocab_size` |
| `n_audio_ctx` | `max_source_positions` (1500) |
| `n_text_ctx` | `max_target_positions` (448) |

Parsed with serde; missing/renamed keys are a `ConfigParse` error.

### 8.3 Safetensors reader (`weights/safetensors.rs`)

Independent reader (~120 lines): 8-byte LE `u64` header length → JSON header of
`name → {dtype, shape, data_offsets}` → raw bytes. F32 only; other dtypes are a
clear error. Tensor bytes reinterpreted as `f32` with length checks.

### 8.4 Name mapping (`weights/loader.rs`)

Checkpoint keys are the original torch names (`encoder.conv1.weight`,
`decoder.blocks.0.attn.query.weight`, `decoder.blocks.0.mlp.0.weight`, …). Rather
than depend on Burn derive path quirks for fields like `mlp.0`, each submodule
exposes:

```rust
fn from_weights<B: Backend>(w: &mut WeightMap<B>, prefix: &str) -> Result<Self>
```

`WeightMap` is a `HashMap<String, Tensor<B, N>>` drained by exact key.
After the whole model is built the map **must be empty** and every expected shape
asserted — turning silent naming/shape bugs into loud load-time errors.

Layout/tying notes that must be honored at load time:

- **Tied embeddings**: the checkpoint has only `decoder.token_embedding.weight`;
  the output logits use it transposed. There is **no** separate output-head key —
  the loader must not expect one.
- **Conv1d layout**: torch stores `[out_channels, in_channels, kernel]`; Burn's
  `Conv1d` weight is the same layout — verify with a load-time shape assert.
- **Dtype**: `model.safetensors` is fp32; `model.fp16.safetensors` (if present) is
  rejected by the F32-only reader with an error naming the file.

### 8.5 Backend selection (`backend.rs`)

No runtime backend enum. Backend is selected at **compile time via features**
(see §4.2): `wgpu` (default) or `ndarray`. `backend.rs` re-exports the two
backends, provides the `WgpuWhisper` / `CpuWhisper` aliases, and a
`default_device()` helper. Rationale: Burn's `Backend` trait is too large to
delegate through a hand-written enum, and generics keep all model/decoding code
backend-agnostic.

## 9. Decoding (full fidelity)

Port of `whisper/decoding.py` + `whisper/transcribe.py`. **v1 scope** covers the
deterministic path; stochastic sampling is explicitly deferred:

- **In v1**: initial tokens (`sot` + language + task + optional `notimestamps`);
  language detection; no-speech detection; greedy (`temperature = 0`) and beam
  search (length penalty + no-repeat filters); timestamps; 30 s chunking with
  `condition_on_previous_text` / `startofprev`.
- **Deferred to v2** (architecture leaves a hook in `TranscribeOptions`):
  temperature-fallback's stochastic branch — `best_of` + multinomial sampling
  (`Tensor::multinomial`) and the compression-ratio trigger (needs zlib). v1
  ships no stochastic sampling and defaults to `temperature = 0`, so the v1
  deterministic path is fully faithful.

## 10. Testing strategy — draft

- **Unit**: safetensors header parsing (fixture bytes); mel filterbank vs golden
  constants; tokenizer encode/decode round-trip + special ids; each model op
  (attention masking, layernorm, gelu) vs captured torch values; frame math.
- **Golden features**: log-mel of a fixture clip vs reference vector.
- **Integration** (feature-gated / `#[ignore]` by default — downloads ~150 MB):
  load `whisper-tiny` from safetensors and transcribe a fixture audio with
  `temperature=0`, greedy, fixed language; compare token ids/text to a reference
  transcription captured once from `openai/whisper`. A small committed fixture
  covers the safetensors reader and tokenizer without network access.
- **Config**: assert parsed dims for tiny/base/small/large-v3/turbo.
- Where possible, reference fixtures are generated once, outside the repo runtime,
  and committed as test data.

## 11. Error handling — draft

`error.rs` defines an `Error` enum (thiserror-style): `Io`, `Download`, `AudioDecode`,
`Resample`, `UnsupportedFormat` (non-F32 safetensors / missing file),
`ConfigParse`, `MissingWeight`/`UnexpectedWeight`/`ShapeMismatch`,
`Decode`, `Tokenizer`. The lib returns `Result<T, Error>` throughout.

## 12. Dependencies (planned)

- `burn` (+ `burn-wgpu`, `burn-ndarray`) — model + tensor ops.
- `ureq` — model download (blocking, minimal).
- `symphonia` — audio decode (all common formats).
- `rubato` (or small home-grown resampler) — resample to 16 kHz.
- `clap` — CLI.
- `thiserror` — error type.
- `rustfft` — optional, only if DFT proves too slow.
- `serde` + `serde_json` — parse `config.json` and the safetensors header.
- `flate2` (zlib) — **v2 only**, compression-ratio fallback trigger.
- `rand` — **v2 only**, stochastic sampling in temperature fallback.

## 13. Remaining to settle before implementation

- Sections 9–11 are drafts pending explicit review.
- Extract and commit the official mel filterbanks as static assets (§7.1).
- Verify `model.safetensors` availability for each HF repo (v1 blocker).
- Decide STFT DFT vs `rustfft` after a quick benchmark.
- Reference test fixtures (audio clip + expected tokens) capture process.
