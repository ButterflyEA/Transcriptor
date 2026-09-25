# Transcriptor — Tauri desktop app design

Date: 2026-09-22
Status: approved design (sections 1–5 reviewed by the user)

## Overview

Transcriptor is a desktop transcription app built on the existing `whisper-burn`
library. Users pick an audio file, set the same parameters the CLI exposes
(backgrounded with sensible defaults), watch the pipeline stream live into a
status panel, and read the timestamped transcript in a text panel. The
transcript can be copied or exported as txt, srt, vtt, json, docx, and pdf.

Approach approved by the user: **hardened core, thin shell** —
`whisper-burn` gains the progress/streaming/cancellation/format machinery
(shared with the CLI), and the app is a thin Tauri 2 wrapper over it.

## Goals

- A modern, comfortable desktop UI (React + Tailwind) for transcribing audio.
- Same capabilities and defaults as `whisper-burn-cli`, exposed as form
  controls: model (tiny → large-v3-turbo → ivrit-hebrew), language
  (auto + full list), task (transcribe/translate), beam size, initial prompt,
  device (wgpu with CPU fallback). Defaults match the CLI: tiny model, greedy
  (beam 0), wgpu, transcribe.
- Live application state display equivalent to the CLI `--progress` flag:
  model download/merge, audio decode, per-window decode, and segments as they
  are produced.
- Export the transcript to txt, srt, vtt, json (already supported today) plus
  docx and pdf.
- Windows-first build/packaging (NSIS + MSI) with no platform-specific code so
  Linux/macOS follows later without rework.

## Non-goals (this phase)

- Full automated UI end-to-end tests (manual `tauri dev` verification instead).
- Cloud/upload features, audio recording, batch/multiple files.
- Live "type-along" word-level timestamps; segment granularity is sufficient.
- macOS/Linux installers (however, no Windows-only APIs are used).

## Architecture

Three crates in the workspace:

1. `crates/whisper-burn` — library. Gains progress/streaming, cancellation,
   download progress, the shared `format` module, and the turbo translation
   validation.
2. `crates/whisper-burn-cli` — binary. Keeps its interface; imports the
   format module from the library instead of its own `output.rs`.
3. `crates/transcriptor/src-tauri` — new Cargo crate (package `transcriptor`).
   Tauri 2 Rust backend (commands + events) with the React UI beside it at
   `crates/transcriptor/ui/`.

The workspace root `Cargo.toml` gains `crates/transcriptor/src-tauri` as a
member and `docx-rs`/`printpdf` workspace dependencies (versions pinned at M1).

## 1. whisper-burn core enhancements

### Progress & streaming channel

`TranscriptionOptions` gains an optional typed callback:

```rust
pub enum StageKind { Download, Weights, Audio, Decode, Done }

pub enum ProgressUpdate {
    /// Pipeline stage transition with a message (e.g. "downloading
    /// ivrit-ai/whisper-large-v3", "decode window 3/4").
    Stage { kind: StageKind, message: String },
    /// One finished segment, emitted as it is decoded.
    Segment(TranscriptionSegment),
}
```

The callback is invoked on the decoding thread from the hook the CLI already
uses for progress logging (`crates/whisper-burn/src/transcribe.rs`, the
segment-progress site). Everything is `Send + Sync` so a Tauri command can
re-emit it as an event.

### Best-effort cancellation

`TranscriptionOptions.cancelled: Arc<AtomicBool>`. The decoder checks it
between windows and aborts cleanly, returning the segments produced so far.
The CLI does not set it (behavior unchanged).

### Download progress

`download_to`/`download_checkpoint` accept an optional progress callback
(bytes + total when Content-Length is available). Covers the single-file and
the sharded ivrit merge path. `Whisper::from_pretrained` gains a
`with_progress` variant so the app can surface model downloads.

### Turbo translation validation moved to the library

The CLI's fast-fail rule (`large-v3-turbo` + `translate`) moves from
`main.rs` into library `validate_options`, so the app cannot bypass it. The
CLI's existing test stays green and the library gains its own test.

### Shared `format` module

Move `whisper-burn-cli/src/output.rs` into `whisper_burn::format`:

- `format_txt`, `format_srt`, `format_vtt`, `format_json` — unchanged content.
- `bracket_line` — console visual-order rendering (unchanged).
- `format_docx(segments, include_timestamps) -> Vec<u8>` — via `docx-rs`.
- `format_pdf(segments, include_timestamps) -> Vec<u8>` — via `printpdf`.

`include_timestamps` toggles the `[00:00.000 --> 00:02.500]` prefix in the
docx/pdf text. The CLI re-imports these functions; its `tests/output.rs`
suites are repointed at `whisper_burn::format` and stay green.

### Public language list

Expose the supported language codes for the app dropdown (auto + language
list) through the existing `decoding::lang` API.

### Known risk: RTL in docx/pdf

Hebrew rendering inside docx/pdf depends on `docx-rs`/`printpdf` bidi support.
Fallback is to embed direction marks / unambiguous direction markers per
segment. Verified during M1 against the Hebrew test clip.

## 2. App structure

- Cargo workspace member: `crates/transcriptor/src-tauri` (package
  `transcriptor`).
- Frontend: `crates/transcriptor/ui/` — React + TypeScript + Vite + Tailwind;
  Vite's dist is bundled by `cargo tauri build`. Node toolchain needed only to
  build/dev the UI.
- Rust dependencies: `whisper-burn` (features `audio`, `weights`), `burn`
  (`wgpu` + `ndarray`), `tauri` 2, `tauri-plugin-dialog` (open + save), serde.
- Load-bearing work (decode + inference) runs on a background `std::thread`;
  the progress callback is forwarded to the React UI via Tauri events.
- One transcription at a time. A guarded `AppState` holds the cancel handle;
  the model/params controls stay interactive during a run, only Transcribe is
  disabled.

### Commands (Rust)

- `select_audio` — dialog pick; returns path + display name + size.
- `defaults` — model list, language list, current defaults, device list.
- `transcribe(params)` — validates, spawns the run, returns immediately; the
  long-running work streams `progress`/`segment` events and finishes with a
  `done` event.
- `stop` — flips the cancellation flag.
- `save_transcript(format, include_timestamps)` — renders via the library and
  writes the file through the native save dialog.
- `copy_transcript(include_timestamps)` — clipboard via the web API
  (`navigator.clipboard`, WebView2) with a fallback for denial.

### Events (Rust → React)

- `progress` — `ProgressUpdate::Stage` with kind/message and download fraction.
- `segment` — streamed `TranscriptionSegment`.
- `done` — final status (finished / cancelled-with-partial).

## 3. UI layout & UX

Single window; Tailwind; light/dark follows system with a manual toggle.

```
+--------------------------------------------------------------+
|  Transcriptor                    [dark toggle]               |
+------------------+-------------------------------------------+
|  PARAMS (rail)   |  STATUS (top of main, collapsible)        |
|  Audio file  [Browse]         weights: fetching ivrit-ai...  |
|  name.wav (12.4s)              [download bar 62% shard 2/2]  |
|  ----------------|  audio: 12.4 s @ 16 kHz  decode 3/4       |
|  Model  [large-v3 v]          [streamed segment lines]       |
|  Language [auto v]  +----------------------------------------+
|  Task   (o) transcribe        |  TRANSCRIPT  [Copy] [Save v] |
|         ( ) translate         |  [00:00.000 --> 00:05.000]   |
|  Beam size [ 0 ]              |      segment text            |
|  Initial prompt [________]    |  ...                         |
|  Device  [wgpu v]             |                              |
|  ----------------             |                              |
|      [Transcribe]  [Stop]     |                              |
+------------------+-------------------------------------------+
```

- Calm palette: neutral background, single accent for primary
  Transcribe/Save, soft cards, generous spacing, system font stack.
- Status panel streams the same stages as `--progress`; segments appear in the
  transcript live.
- Save opens a format picker (txt/srt/vtt/json/docx/pdf) + timestamps toggle
  for docx/pdf, then the OS save dialog.
- Empty states with guidance; errors surface inline in the status panel;
  Hebrew renders in logical order in-app (visual-order mirroring remains a
  console-only concern).

## 4. Data flow & errors

Happy path: browse → transcribe (validate → device pick → load/download model
→ decode → per-window decode with progress) → segments stream → done event →
save/copy.

Errors:
- All library `Error`s map to serializable `AppError { kind, message }`
  rendered inline in the status panel.
- A second `transcribe` while one runs is rejected (and disabled in the UI).
- `stop` keeps the partial transcript; status reports "cancelled".
- Download failures produce plain-language guidance; unsupported audio shows
  the decode error.
- Turbo+translate is rejected by shared `validate_options` before any thread
  starts.

## 5. Testing & milestones

**Rust TDD everywhere** for M1 and M3–M5, with the regression gate
(workspace suite + clippy) green at every milestone:
- Library: progress-event ordering; cancellation mid-stream; turbo rule in
  `validate_options`; docx/pdf renderers (real bytes; docx read back via
  zip/XML; pdf structural validity + transcript text presence); format
  extraction keeps every existing CLI/output test green.
- App crate: `save_transcript` writes correct bytes per format; `AppError`
  mapping; second-run guard.
- React: manual verification in `tauri dev`; minimal vitest only for pure
  helpers; full UI E2E deferred.
- Windows: `cargo tauri build` produces NSIS + MSI; installer smoke-tested.

**Milestones:**

1. **M1** — whisper-burn enhancements + `format` extraction with docx/pdf;
   CLI stays green.
2. **M2** — scaffold `crates/transcriptor/src-tauri` + `ui/`; workspace
   wiring; blank window renders under `tauri dev`.
3. **M3** — commands/events; params rail; browse/transcribe streaming into
   the UI.
4. **M4** — status/progress panel, Stop, inline error mapping.
5. **M5** — save/export all six formats + copy; dark/light; polish.
6. **M6** — Windows installer build + smoke test + manual
   Hebrew/English/mp3/translate/cancel pass.

## Phase 2 (later)

- Linux/macOS installer config (no code changes expected — no Windows-only
  APIs are used).
- Full UI E2E automation.
- Additional output niceties per demand.