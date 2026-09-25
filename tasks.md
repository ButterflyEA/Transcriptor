# Whisper-Burn Implementation Tasks

Full implementation plan: `docs/superpowers/plans/2026-09-19-whisper-burn.md`
Design spec: `docs/superpowers/specs/2026-09-19-whisper-burn-design.md`

---

## T1: Workspace scaffold ✅

**User story:** As a maintainer, I want a Cargo workspace with a library crate and a CLI crate so the model can be reused and the CLI can stay thin.

- [x] Convert root `Cargo.toml` to a pure `[workspace]` manifest
- [x] Create `crates/whisper-burn` lib crate with `Cargo.toml` (burn 0.21, thiserror, serde, serde_json, base64, fancy-regex; optional wgpu/ndarray/audio/weights)
- [x] Create `crates/whisper-burn-cli` bin crate with clap
- [x] Configure features: `default = ["wgpu", "ndarray"]`, `wgpu`, `ndarray`, `audio`, `weights` — **wgpu is the default backend** (deviation from plan: `ndarray` added to default so the runtime CPU fallback in T17 can compile without reselecting features)
- [x] Requirement (from planning): default device = **wgpu**; when no GPU is available at runtime, **fall back to CPU/ndarray** instead of failing (see T17)
- [x] Add `src/lib.rs` `version()` stub + `crates/whisper-burn-cli/src/main.rs` so `cargo build --workspace` resolves
- [x] Verify `cargo build --workspace`, `cargo test --workspace`, `cargo run -p whisper-burn-cli` (prints version); existing T2–T10 suites green; 3 failures remain only in uncommitted T11 WIP tests (`loader.rs`, `ops.rs`)

---

## T2: Error type and Result alias ✅

**User story:** As an API consumer, I want a typed error enum so every failure is descriptive and matchable instead of a raw `String`.

- [x] Define `Error` enum variants (Io, Download, MissingWeights, UnsupportedFormat, ConfigParse, MissingWeight, UnexpectedWeight, ShapeMismatch, Decoder, Tokenizer, AudioDecode, Resample)
- [x] Add `Result<T>` alias
- [x] Derive `thiserror::Error` + `Debug`
- [x] Unit test constructing/formatting several variants

---

## T3: ModelDimensions, config.json parsing, ModelSize ✅

**User story:** As a user, I want the architecture read from each checkpoint's `config.json` so no model size is hardcoded.

- [x] Define `ModelDimensions` struct (n_mels, n_vocab, n_audio_layer, n_text_layer, n_audio_state, n_text_state, n_head, n_audio_ctx, n_text_ctx)
- [x] Define `HfConfig` serde struct + key mapping (num_mel_bins, encoder_layers, d_model, encoder_attention_heads, max_source_positions, max_target_positions, ...)
- [x] Define `ModelSize` enum with repo ids + expected-dims checks
- [x] Implement `read_config` + `ModelSize::validate`
- [x] Add `tests/fixtures/tiny_config.json` fixture
- [x] Unit tests for parsing + validation (wrong n_mels → error)

---

## T4: Safetensors reader ✅

**User story:** As an engineer, I want a reliable F32 safetensors reader so weights load exactly as stored.

- [x] Define `TensorMeta` (dtype, shape) + `SafeTensors` container
- [x] Parse header JSON (length-prefixed) + `data_start` from end of header
- [x] Implement `read_safetensors` with overflow/truncation checks
- [x] Implement `slice_f32` (F32 only; 4-alignment checked, no unchecked casts)
- [x] Add synthetic-file unit tests (valid, F16 rejected, truncated header)

---

## T5: Tokenizer core (tiktoken byte-level BPE) ✅

**User story:** As an engineer, I want byte-level BPE encoding/decoding that matches the reference tiktoken so token ids are identical.

- [x] Define `CoreBpe` + `.tiktoken` file parsing (`<base64> <rank>`)
- [x] Add `pat_str` regex for byte-pair tokenization
- [x] Implement `byte_pair_merge` + `encode_ordinary`
- [x] Implement `decode`
- [x] Add `tests/fixtures/mini.tiktoken` fixture
- [x] Unit tests for roundtrip + ranks

---

## T6: Whisper tokenizer (special tokens, languages, timestamps) ✅

**User story:** As a user, I want whisper's special/language/timestamp tokens available so decoding matches the reference protocol.

- [x] Define `TextTokenizer` with special ids (eot, sot, translate, transcribe, startoflm, startofprev, nospeech, notimestamps, timestamp_begin)
- [x] Add `LANGUAGES` table + `language_token()` / code lookup
- [x] Implement `timestamp_token(t)` = timestamp_begin + t
- [x] Implement `sot_sequence` (sot, language, task, no_timestamps)
- [x] Handle `yue` appended at `n_vocab - 1` when `n_vocab == 51866`
- [x] Unit tests for languages, timestamps, sequence construction

---

## T7: STFT (torch center=True semantics) ✅

**User story:** As an engineer, I want a DFT STFT producing power spectra that matches torch's centered STFT.

- [x] Define `StftConfig` (n_fft=400, hop=160) + `features/mod.rs`
- [x] Implement Hann window (periodic, matches `torch.hann_window` default)
- [x] Implement `dft_power` (|X|² for 0..=n_fft/2)
- [x] Implement `stft_power` with n_fft/2 symmetric zero padding (n_frames = len/hop + 1)
- [x] Golden small-case test (hand-computed n_fft=4) + sine-peak test

---

## T8: Mel filterbank assets + log-mel transform ✅

**User story:** As a user, I want log-mel features identical to the reference so transcription quality matches.

- [x] Write `scripts/extract_mel_filters.py` + generate `assets/mel_filters_80.bin` / `mel_filters_128.bin` (f32 row-major)
- [x] Implement `FeatureExtractor::new(n_mels, sample_rate)` loading baked assets
- [x] Implement `log_mel` (stft → power → filter matmul → log10 clamp 1e-10 → max−8 clip → (x+4)/4)
- [x] Implement `window.rs` chunking (`N_SAMPLES=480000`, `N_FRAMES=3000`, `split_chunks`)
- [x] Tests: asset shapes, chunk padding, log-mel shape/math

---

## T9: Audio decode (symphonia) ✅

**User story:** As a user, I want common audio formats decoded to mono f32 so I can transcribe any file.

- [x] Implement `decode_to_mono_f32` via symphonia (probe, decode packets, convert to f32)
- [x] Downmix channels to mono (average)
- [x] Extract source sample rate
- [x] WAV round-trip test fixture + golden values
- [x] deps bumped to latest crates.io: base64 0.23.1, fancy-regex 0.19.2, symphonia 0.6.1

---

## T10: Resampler to 16 kHz ✅

**User story:** As a user, I want audio auto-resampled to 16 kHz so no manual preprocessing is needed.

- [x] Implement `resample_to_16k` via rubato (5.0.0) `Async::new_sinc` + `process_all`
- [x] No-op when input is already 16 kHz
- [x] Tests: no-op, 8k→16k length ~2x, sine energy preserved

---

## T11: WeightMap + custom Linear / LayerNorm / GELU(erf)

**User story:** As an engineer, I want a weight loader that guarantees every checkpoint key is either loaded or an error, and primitive layers that match the reference math.

- [x] Implement `WeightMap` (new, take_1d/2d/3d, finish, consumed tracking)
- [x] Implement `erf` (A&S 7.1.26) and `gelu_erf`
- [x] Implement `Linear` forward (`x @ Wᵀ + b`)
- [x] Implement `LayerNorm` forward (mean/var over last dim, eps 1e-5)
- [x] Register `weights::loader` in `weights/mod.rs` + crate mods
- [x] Tests: WeightMap drain/leftover/shape-mismatch; erf reference values; linear matmul golden; layer-norm golden
- [x] Notes: `mean_dim` in burn 0.21 keeps rank (dim → 1), so no explicit unsqueeze is needed; `slice_f32` now computes the 8-byte-padded `data_start` per the safetensors spec (matches the already-padded test files)

---

## T12: MultiHeadAttention (scaled dot-product, causal + KV cross)

**User story:** As an engineer, I want attention that numerically matches the reference q·kᵀ/√d_head softmax.

- [x] Define `MultiHeadAttention` (query, key, value, out Linears, n_head)
- [x] Reshape QKV to heads and apply `1/√d_head` scale on K
- [x] Implement softmax over keys with additive mask
- [x] Implement output projection
- [x] Tests: single-head identity math (self + cross) validated by hand, multi-head split/concat, causal mask
- [x] Notes: `mask` is rank-4 `[1,1,q,k]` (matches `causal_mask` in the plan's decoder); scale folded into `kᵀ` via `mul_scalar` (0.21 has no `FloatElem::from_f64`)

---

## T13: ResidualAttentionBlock

**User story:** As an engineer, I want the residual block with optional cross-attention matching the reference layer order.

- [x] Define `ResidualAttentionBlock` fields (attn, attn_ln, mlp, mlp_ln, mlp2, ln, xa_attn, xa_attn_ln, xa_ln)
- [x] Forward: self-attn → cross-attn (query-side `xa_attn_ln`, KV = raw `xa`) → mlp2(gelu(mlp))
- [x] Keep `xa_ln` present-but-unused (consumed for weight drain)
- [x] Tests: identity-path single token; cross-attention uses KV + query LN

---

## T14: AudioEncoder (conv1d + blocks + positional)

**User story:** As an engineer, I want the audio encoder producing 1500 token features exactly like the reference.

- [x] Implement `Conv1d` forward (symmetric pad, per-offset matmuls, stride-2 via even-index selection)
- [x] Define `Encoder` (conv1, conv2, positional_embedding [1500, n_state], blocks, ln_post)
- [x] Forward: gelu(conv1) → gelu(conv2) → transpose → +pos → blocks → ln_post
- [x] Tests: conv vs naive reference (stride 1 and 2); encoder wiring vs manual pipeline (composed from independent golden-tested parts; note: plan's "all ones" golden is wrong — identity LN zeroes a constant input)

---

## T15: TextDecoder (tied embeddings, causal mask, logits)

**User story:** As an engineer, I want the text decoder producing tied logits over the vocabulary.

- [x] Define `Decoder` (token_embedding [n_vocab, n_state], positional_embedding [n_text_ctx, n_state], blocks, ln)
- [x] Build causal mask (upper-triangle -inf)
- [x] Forward: embed + pos → blocks with cross-attn + mask → ln → `x @ token_embeddingᵀ`
- [x] Tests: causal mask shape/content golden; zero-weight decoder tied-logits hand golden (with correct identity-LN math — the plan's "embed row" values ignored LN); wiring + cross/causal sensitivity test

---

## T16: Whisper model assembly + strict WeightMap drain

**User story:** As an engineer, I want the full encoder+decoder assembled from weights with no unconsumed keys.

- [x] Implement weight-loading helpers (`ln`, `linear`, `mha`, `block`) + encoder/decoder builders
- [x] Define `Whisper` struct (dims, encoder, decoder)
- [x] `from_weights` + `forward_encoder` + `forward_decoder`
- [x] Consume decoder `xa_ln.weight` (bias-less) for drain compliance
- [x] Test: synthetic full tiny checkpoint loads and `finish()` passes
- [x] Notes: `Whisper` uses real tiny-ish dims (headroom for speed); tests generate a complete weight set programmatically (all-ones fill so forwards stay finite), plus missing-key → MissingWeight and wrong-shape → ShapeMismatch paths; drain-guard mutation (skip `xa_ln.weight`) caught by `finish()` with UnexpectedWeight

---

## T17: Backend aliases (wgpu default, runtime CPU fallback)

**User story:** As a user, I want `CpuWhisper` / `WgpuWhisper` type aliases so I never touch backend generics, and the default backend to be wgpu with an automatic CPU fallback when no GPU is present.

- [x] Define `CpuWhisper = Whisper<NdarrayBackend<f32>>` (feature ndarray)
- [x] Define `WgpuWhisper = Whisper<WgpuBackend>` (feature wgpu)
- [x] Add device helpers (`cpu_device`, wgpu adapter handling)
- [x] **Runtime fallback: attempt wgpu (automatic adapter, incl. software/CPU); if no GPU/adapter is available, fall back to the ndarray CPU device instead of erroring**
- [x] Type-level compile test

---

## T18: Model weight download (ureq)

**User story:** As a user, I want checkpoints and configs auto-downloaded from Hugging Face.

- [x] Implement `repo_base` / repo id for each `ModelSize`
- [x] Implement atomic `download_to` (`.part` + rename, retry ×3)
- [x] Implement `download_checkpoint` (config.json + model.safetensors)
- [x] Implement `cache_dir` (with `WHISPER_BURN_CACHE` override)
- [x] `#[ignore]` network test downloading tiny

---

## T19: Whisper::load / from_pretrained

**User story:** As a user, I want to instantiate a model from a local checkpoint or from the cache.

- [x] Implement `load(size, weights_dir, device)` (validates config, drains weights)
- [x] Implement `from_pretrained(size, device)` (cache + gated download; MissingWeights if disabled)
- [x] Tests: fake checkpoint load; missing weights file; missing single key

---

## T20: forward_mel public + first encoder pass on real checkpoint

**User story:** As an engineer, I want to verify the encoder output against the reference before writing decode logic.

- [x] Expose `forward_encoder` + feature-extraction wiring
- [x] Add `tests/golden_encoder.rs` (`#[ignore]`), run against tiny checkpoint on a 440 Hz tone
- [x] Debug conv layout / mel orientation if encoder drifts from reference
- [x] Commit once encoder numerics are stable

---

## T21: Language detection + no-speech classifier

**User story:** As a user, I want automatic language detection and silence detection.

- [x] Implement `detect_language` (1-step decoder logits, argmax over language-token window)
- [x] Compute language probability (softmax over window)
- [x] `#[ignore]` network test: silence → `en`

---

## T22: Greedy decoding (temperature 0) + no-speech

**User story:** As a user, I want deterministic greedy transcription.

- [x] Define `GreedyOptions` (max_tokens, suppress_blank, suppress_tokens, thresholds)
- [x] Implement `greedy_search` loop (logits → suppress → argmax → push until eot)
- [x] Track `mean_logprob` for no-speech decisions
- [x] Terminate at `max_tokens`
- [x] `#[ignore]` network test: silence terminates quickly

---

## T23: Beam search with timestamps (deterministic path)

**User story:** As a user, I want beam search to improve transcription accuracy.

- [x] Define `BeamOptions` (beam_size, best_of, patience, max_tokens, suppression)
- [x] Implement `beam_search` (top-k per beam, finished lists, patience)
- [x] Ensure beam_size=1 agrees exactly with greedy
- [x] `#[ignore]` network test

---

## T24: Timestamps + segment assembly (reference format)

**User story:** As a user, I want timestamped text segments with the same format as the reference.

- [x] Implement `timestamp_token_to_ms` (×10 ms)
- [x] Implement `assemble_segments` (parse timestamps vs text runs into Segments)
- [x] Unit tests: ms math; segment splitting/end-times; eot handling

---

## T25: Whisper::transcribe orchestration (full pipeline)

**User story:** As a user, I want one `transcribe(pcm, sample_rate, options)` call that returns timestamped text for any audio length.

- [x] Define `TranscriptionOptions` (language, task, beam_size, temperature(0 only in v1), thresholds, initial_prompt)
- [x] Chunk loop: resample → 30 s chunks → mel → encode → detect language → decode → segments
- [x] Apply no-speech filtering (mean_logprob < threshold)
- [x] Implement `condition_on_previous_text` cross-chunk prefix
- [x] `#[ignore]` network integration test on a short real clip

---

## T26: CLI (whisper-burn binary)

**User story:** As a user, I want to transcribe audio from the command line.

- [x] Parse clap args (model, language, task, beam-size, device, output format)
- [x] Select backend (wgpu default with automatic CPU fallback when unavailable, `--device cpu` forces ndarray)
- [x] Wire: download/load → decode audio → transcribe → print segments
- [x] Write txt / srt / vtt / json outputs
- [x] Integration smoke test (`#[ignore]`)

---

## T26a: F16/BF16 safetensors weights

**User story:** As a user, I want every whisper model size to load, including
the large checkpoints (`medium`, `large`, `large-v2`, `large-v3`,
`large-v3-turbo`) which HF publishes with F16 (and BF16) tensors.

- [x] `slice_f32` → `Cow<[f32]>`: F32 zero-copy borrow; F16/BF16 converted to f32 on read
- [x] IEEE half / bfloat16 → f32 bit conversions (denormals, inf/nan preserved)
- [x] Update T4's "F16 rejected" test → F16/BF16 round-trip tests
- [x] Verified vs numpy reference (`0x3C00→1.0`, `0x7BFF→65504.0`, subnormal `2^-24`, ±inf)
- [x] End-to-end: `--model large-v3` on the real cached checkpoint (1259 F16 tensors) transcribes

---

## T26b: `--progress` pipeline logging

**User story:** As a user, I want to see each pipeline stage (model loading,
audio conversion, language detection, per-window decode, finished segments)
streamed to stderr while the transcript goes to stdout.

- [x] Add `log` crate to workspace; library emits `log::info!` stage lines (no-op without a logger)
- [x] `download_checkpoint` + `from_pretrained`/`load`: cached/fetch, file size, model dims, tensor count
- [x] `transcribe`: window count, per-window range, language detection, decode result (tokens, no-speech), streamed segment lines
- [x] CLI `--progress` flag installs a minimal stderr logger (`[elapsed] message`); wgpu/burn/cubecl noise filtered by log target; `--verbose` unchanged
- [x] CLI logs audio decode + `wrote <path>`
- [x] Tests: help lists `--progress`; `#[ignore]` integration test asserts stage + streamed segment lines on stderr
- [x] Validated live on the 22 s wav: `--progress --model tiny` shows weights → model → decode → language `he` → segment

---

## T27: Acceptance — cement goldens vs reference whisper

**User story:** As a maintainer, I want proof that our output matches the official whisper implementation.

- [ ] Write `scripts/cement_reference.py` (official openai/whisper → golden fixtures for jfk.flac)
- [ ] Log-mel allclose (atol 1e-4)
- [ ] Encoder allclose (atol 1e-3)
- [ ] Greedy token ids exactly equal
- [ ] Beam output equal (beam 4, patience 1)
- [ ] Segment text equality to `expected_segments.txt`
- [ ] Final gates: offline suite green, golden suite green, clippy clean
- [ ] Commit golden fixtures + fix commits