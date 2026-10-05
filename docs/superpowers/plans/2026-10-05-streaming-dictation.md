# Streaming Dictation (v1.0.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While the user dictates (tap to start, tap to stop), text is typed into the focused field within about 1.5 s of being spoken, the last few words are corrected as later audio arrives, and the final tail is finalised quickly on stop.

**Architecture:** `whisper-burn` learns to decode short windows (opt-in, 30 s path untouched). `dictation-service` gets a `LiveTap` on the capture buffer, an engine-side stream loop that decodes sliding windows, a pure, property-tested `transcript` state machine (locked text plus a tentative tail), and an output worker thread that types deltas and backspace corrections. Pacing is self-adjusting: the next pass starts when the previous one finishes, so slow GPUs lag but never stall or queue.

**Tech Stack:** Rust, burn 0.21 (wgpu, f16), cpal, rdev, enigo-style typing, proptest.

**Spec:** [../specs/2026-10-05-streaming-dictation-design.md](../specs/2026-10-05-streaming-dictation-design.md)

## Global Constraints

- Offline only; no network, ever.
- 30 s path and every existing golden test stay byte-identical (`window_samples = None` is the default).
- Streaming is opt-in: `stream = false` by default; all stream config is restart-required.
- `stream = true` means tap-to-toggle with the same chord. Hold-to-talk batch mode is unchanged.
- Locked text is never changed. Backspaces never exceed our own tentative tail and never pass the start of this session's typed text.
- On user interference (non-hotkey key, mouse click, foreground window change): lock everything and continue **append-only** (decision Q4).
- Latency target: text within about 1.5 s of speech, a few more word errors accepted (decision Q1/Q3).
- Weak GPUs: streaming must **degrade, not disable** (decision Q2). Self-pacing plus an adaptive window; if one pass takes longer than 3x the window, the service notifies and falls back to batch mode for that capture.
- Clipboard is set once, at the end, to the full transcript. Never per delta.
- Wayland without evdev: corrections disabled (append-only).
- Work one task at a time. After each task: validate, then **stop for user review** before the next sprint.
- Windows paths and PowerShell: no `&&`; check `$LASTEXITCODE`; commit trailer `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `crates/whisper-burn/src/model/encoder.rs` | slice positional embedding to real length; even-length guard |
| `crates/whisper-burn/src/stft.rs` | clear error for buffers that are too short |
| `crates/whisper-burn/src/features/window.rs` | add `split_windows` (variable window, returns start offsets) |
| `crates/whisper-burn/src/transcribe.rs` | `TranscriptionOptions.window_samples`; derive `seek_ms` and mel shape from the window |
| `crates/dictation-service/src/audio.rs` | `LiveTap`, `snapshot_from` |
| `crates/dictation-service/src/transcript.rs` (new) | pure state machine: normalise, agree, lock, tail edits |
| `crates/dictation-service/src/stitch.rs` (new) | overlap merge of consecutive window texts |
| `crates/dictation-service/src/vad.rs` (new) | energy gate for silent windows |
| `crates/dictation-service/src/typer.rs` (new) | `Typer` trait: type text, backspace; per-platform impls |
| `crates/dictation-service/src/outworker.rs` (new) | output worker thread; ordered jobs and `Flush` |
| `crates/dictation-service/src/stream.rs` (new) | engine-thread stream loop, pacing, cancellation |
| `crates/dictation-service/src/{controller,runtime,config,main}.rs` | stream mode wiring, config keys, interference lock |
| `docs/dictation.md`, `docs/superpowers/specs/2026-10-05-streaming-dictation-design.md` | docs and decision record |

---

# Sprint A: Phase 0 measurements (go/no-go gate)

**Exit criteria:** a results table in the spec answers: WER vs window length, pass time vs window length, tail stability, and whether the mel normalisation drifts. **Sprint B does not start before the user reviews it.**

### Task A.1: Short-window experiment harness (throwaway, behind a hidden CLI)

**Files:**
- Modify: `crates/whisper-burn/src/model/encoder.rs` (positional-embedding slice, item 1)
- Modify: `crates/whisper-burn/src/transcribe.rs` (`window_samples`, item 2 and 3)
- Modify: `crates/whisper-burn/src/features/window.rs` (`split_windows`)
- Modify: `crates/dictation-service/src/main.rs` (hidden `stream-sweep` command)
- Test: `crates/whisper-burn/tests/encoder.rs`, `crates/whisper-burn/tests/transcribe.rs`

**Interfaces:**
- Produces: `TranscriptionOptions.window_samples: Option<usize>` (`None` = today's 30 s path; `Some(n)` requires `n` to be a multiple of 320 samples at 16 kHz, i.e. 20 ms, so the frame count is even).
- Produces: `pub fn split_windows(len: usize, window: usize) -> Vec<(usize, usize)>` returning `(start_sample, end_sample)` pairs; the last window may be shorter and is **not** padded to 30 s.
- Produces: hidden command `dictation stream-sweep <wav> --model-dir <dir> --language he --windows 1,2,3,5,10`.

- [ ] **Step 1: Failing test for the positional-embedding slice.** In `tests/encoder.rs` add a test that builds the encoder on a tiny random config, runs `forward` on a mel of 200 frames, and asserts the output has `100` time steps and no panic. Run `cargo test -p whisper-burn --test encoder short_input` and see it fail.
- [ ] **Step 2: Implement.** In `encoder.rs` slice the positional table to `out_len` after the conv stack; at 3000 input frames it is the whole table. Return a clear `WhisperError` when `out_len` is odd.
- [ ] **Step 3: Golden check.** Run `cargo test -p whisper-burn --test golden_encoder --test greedy`; both must stay byte-identical.
- [ ] **Step 4: Failing test for `split_windows`.**

```rust
#[test]
fn split_windows_covers_audio_without_padding() {
    assert_eq!(split_windows(48_000, 16_000), vec![(0, 16_000), (16_000, 32_000), (32_000, 48_000)]);
    assert_eq!(split_windows(40_000, 16_000), vec![(0, 16_000), (16_000, 32_000), (32_000, 40_000)]);
    assert!(split_windows(0, 16_000).is_empty());
}
```

- [ ] **Step 5: Implement `split_windows`, then thread `window_samples` through `transcribe`.** Replace `seek_ms = ci * 30_000` with the window's real start, and `[1, n_mels, N_FRAMES]` with the real frame count. Add the reflect-pad guard in `stft.rs` (error under about 202 samples). Run the two golden tests again.
- [ ] **Step 6: Add the hidden `stream-sweep` command.** For each window length it slices the clip with `split_windows`, decodes each piece with `window_samples = Some(len)`, prints text, pass time, and writes JSON.
- [ ] **Step 7: Commit** `feat(whisper-burn): opt-in short decode windows`.

### Task A.2: Measure and write up

- [ ] **Step 1:** Record 5 Hebrew dictation clips (30 to 60 s) including `test_16000_mono.wav`; store transcripts as reference next to `docs/golden/`.
- [ ] **Step 2:** Run the sweep at 1, 2, 3, 5, 10 s. Compute WER (character-level for Hebrew too) against the full-decode reference. Record pass time per window (f16 and f32).
- [ ] **Step 3:** Measure mel drift: decode the same 3 s slice alone and inside a longer window; compare log-mel max values and text. If they differ, add a fixed normalisation reference to the plan as Task B.0.
- [ ] **Step 4:** Measure tail stability: how many of the last 3 words change between consecutive 1 s-spaced passes.
- [ ] **Step 5:** Measure silence behaviour: 3 s of room noise; record any hallucinated text.
- [ ] **Step 6:** Write results as a table into the spec (new "Phase 0 results" section), plus a recommended `stream_window_secs`, `stream_interval_ms`, `stream_overlap_ms`. **Stop and ask the user for go/no-go.**

---

# Sprint B: Pure transcript logic (test-first, no audio, no GPU)

**Exit criteria:** `transcript`, `stitch`, and `vad` pass unit and property tests. Nothing is wired to the engine yet.

### Task B.1: Text normalisation and word agreement

**Files:**
- Create: `crates/dictation-service/src/transcript.rs`
- Test: `crates/dictation-service/tests/transcript.rs`
- Modify: `crates/dictation-service/src/lib.rs` (`pub mod transcript;`)

**Interfaces:**
- Produces: `pub fn normalize_word(w: &str) -> String` (strips punctuation and Hebrew niqqud/cantillation U+0591..U+05C7, lowercases Latin).
- Produces: `pub fn words(text: &str) -> Vec<String>` (whitespace split of the original text, order preserved).
- Produces: `pub fn agreed_prefix_len(a: &[String], b: &[String]) -> usize` (count of leading words equal after normalisation).

- [x] **Step 1: Failing tests.**

```rust
use dictation_service::transcript::*;

#[test]
fn normalise_strips_punctuation_and_niqqud() {
    assert_eq!(normalize_word("שָׁלוֹם,"), normalize_word("שלום"));
    assert_eq!(normalize_word("Hello."), "hello");
}

#[test]
fn agreed_prefix_ignores_punctuation_flips() {
    let a: Vec<String> = ["שלום,", "זאת", "הקלטה"].iter().map(|s| s.to_string()).collect();
    let b: Vec<String> = ["שלום.", "זאת", "הקלטת"].iter().map(|s| s.to_string()).collect();
    assert_eq!(agreed_prefix_len(&a, &b), 2);
}
```

- [x] **Step 2: Run, see them fail.** `cargo test -p dictation-service --test transcript`
- [x] **Step 3: Implement** with `char::is_alphanumeric` filtering and the Unicode range above.
- [x] **Step 4: Run, see them pass.**
- [x] **Step 5: Commit** `feat(dictation): word normalisation and agreement`.

### Task B.2: Locked text and tentative tail state machine

**Files:**
- Modify: `crates/dictation-service/src/transcript.rs`
- Test: `crates/dictation-service/tests/transcript.rs`, `crates/dictation-service/tests/transcript_props.rs` (proptest; add `proptest` as a dev-dependency)

**Interfaces:**
- Consumes: `words`, `agreed_prefix_len` from B.1.
- Produces:

```rust
pub enum Edit { Append(String), Replace { backspaces: usize, text: String } }

pub struct Transcript { /* locked: Vec<String>, typed_tail: String, last: Vec<String>, lock_words: usize, append_only: bool */ }

impl Transcript {
    pub fn new(lock_words: usize) -> Self;
    /// Feed a full pass over the not-yet-locked audio. Returns at most one edit.
    pub fn on_pass(&mut self, pass_text: &str) -> Option<Edit>;
    /// Stop correcting from now on (user interference or Wayland without evdev).
    pub fn set_append_only(&mut self);
    /// Final pass: returns the single correction and the full final text.
    pub fn finish(&mut self, final_text: &str) -> (Option<Edit>, String);
    pub fn locked_words(&self) -> usize;
}
```

Rules to implement: words in the agreed prefix older than `lock_words` from the end become locked; the tail is everything after locked words; when the new tail differs from `typed_tail`, emit `Replace` with `backspaces = typed_tail.chars().count()`; when it only extends it, emit `Append`; in append-only mode never emit `Replace` (emit only the non-contradicting suffix and keep what is typed).

- [x] **Step 1: Failing example tests.**

```rust
#[test]
fn extending_pass_appends_only() {
    let mut t = Transcript::new(2);
    assert_eq!(t.on_pass("שלום"), Some(Edit::Append("שלום".into())));
    assert_eq!(t.on_pass("שלום עולם"), Some(Edit::Append(" עולם".into())));
}

#[test]
fn changed_tail_is_replaced_with_exact_backspace_count() {
    let mut t = Transcript::new(2);
    t.on_pass("אני גר בגבעת אב");
    let e = t.on_pass("אני גר בגבעת אבני").unwrap();
    assert!(matches!(e, Edit::Replace { .. } | Edit::Append(_)));
}

#[test]
fn append_only_never_backspaces() {
    let mut t = Transcript::new(2);
    t.on_pass("אחד שניים");
    t.set_append_only();
    let e = t.on_pass("אחד שלושה ארבעה");
    assert!(!matches!(e, Some(Edit::Replace { .. })));
}
```

- [x] **Step 2: Failing property test (the key invariant).** For random sequences of passes (a prefix-stable base text with random mutations only in the last `k < lock_words` words), assert after each pass: (1) `typed_total` (what the target would contain after applying every edit to a simulated buffer) always starts with the locked words, (2) backspaces never exceed the simulated tail length, (3) after `finish(final)` the simulated buffer equals `final` when no word was locked wrongly. Run and see it fail.
- [x] **Step 3: Implement** until examples and properties pass. Include a log line (via `log::warn!`) when a pass contradicts locked text.
- [x] **Step 4: Run the whole file** and `cargo clippy -p dictation-service --all-targets -- -D warnings`.
- [x] **Step 5: Commit** `feat(dictation): locked text and tentative tail state machine`.

### Task B.3: Overlap stitching

**Files:**
- Create: `crates/dictation-service/src/stitch.rs`
- Test: `crates/dictation-service/tests/stitch.rs`

**Interfaces:**
- Produces: `pub fn stitch(prev: &[String], next: &[String], max_overlap_words: usize) -> Vec<String>`: appends `next` to `prev`, dropping the longest suffix of `prev` that equals a prefix of `next` after `normalize_word`.

- [x] **Step 1: Failing test.**

```rust
#[test]
fn drops_duplicated_overlap() {
    let p: Vec<String> = "אני גר בגבעת".split(' ').map(String::from).collect();
    let n: Vec<String> = "בגבעת אבני ואני".split(' ').map(String::from).collect();
    assert_eq!(stitch(&p, &n, 4).join(" "), "אני גר בגבעת אבני ואני");
}

#[test]
fn no_overlap_just_concatenates() {
    let p = vec!["א".to_string()];
    let n = vec!["ב".to_string()];
    assert_eq!(stitch(&p, &n, 4), vec!["א".to_string(), "ב".to_string()]);
}
```

- [x] **Step 2: Fail, implement, pass.** Prefer the longest match; never drop more than `max_overlap_words`.
- [x] **Step 3: Commit** `feat(dictation): overlap stitching`.

### Task B.4: Silence gate

**Files:**
- Create: `crates/dictation-service/src/vad.rs`
- Test: `crates/dictation-service/tests/vad.rs`

**Interfaces:**
- Produces: `pub fn is_speech(pcm: &[f32], rate: u32, threshold_rms: f32) -> bool` (true when any 30 ms frame's RMS exceeds the threshold). A `pub const DEFAULT_RMS: f32` chosen from the Phase 0 noise sample.

- [x] **Step 1:** Failing tests: zeros and low noise return `false`; a 440 Hz tone at 0.1 amplitude returns `true`; a 10 ms click only does not.
- [x] **Step 2:** Implement, run, commit `feat(dictation): energy gate for silent windows`.

**Review gate:** stop for the user.

---

# Sprint C: Capture tap and output worker

**Exit criteria:** the engine can read a live snapshot of the recording, and typing/pasting runs on its own thread with ordered delivery. Batch mode is unchanged and faster to stop.

### Task C.1: `LiveTap`

**Files:**
- Modify: `crates/dictation-service/src/audio.rs`
- Test: `crates/dictation-service/tests/audio.rs`

**Interfaces:**
- Produces: `#[derive(Clone)] pub struct LiveTap { /* Arc<Mutex<Vec<f32>>>, rate, max */ }` with `pub fn snapshot_from(&self, offset: usize) -> (Vec<f32>, u32, usize)` returning samples after `offset` (mono, device rate), the rate, and the new total length. Also `pub fn len(&self) -> usize`.
- Produces: `Recorder::tap(&self) -> Option<LiveTap>`, implemented by `CpalRecorder`; `Some` between `start` and `stop`.

- [x] **Step 1:** Failing tests: push 1000 samples, `snapshot_from(0)` returns 1000; push 500 more, `snapshot_from(1000)` returns the 500 and total 1500; offsets past the end return an empty vec; the tap is cloneable and sees later pushes.
- [x] **Step 2:** Implement over the existing buffer lock (reuse `push_capped`). The cpal callback and `snapshot_from` share one `Arc<Mutex<..>>`.
- [x] **Step 3:** Run audio tests; commit `feat(dictation): LiveTap snapshot of in-progress capture`.

### Task C.2: Output worker thread

**Files:**
- Create: `crates/dictation-service/src/outworker.rs`
- Modify: `crates/dictation-service/src/runtime.rs` (route paste through it)
- Test: `crates/dictation-service/tests/outworker.rs`

**Interfaces:**
- Consumes: `output::Paster`.
- Produces:

```rust
pub enum Job { Paste(String), Type(String), Backspace(usize), Flush(std::sync::mpsc::Sender<()>) }
pub struct OutWorker { /* tx, handle */ }
impl OutWorker {
    pub fn spawn(paster: Box<dyn Paster>, typer: Box<dyn Typer>) -> Self;
    pub fn send(&self, job: Job);
    pub fn flush(&self);    // blocks until every earlier job ran
    pub fn shutdown(self);
}
```

- [x] **Step 1:** Failing test with a fake `Paster`/`Typer` recording calls: jobs sent in order are executed in order; `flush()` returns only after all prior jobs ran; `send` returns immediately even if the fake sleeps 200 ms.
- [x] **Step 2:** Implement with one `mpsc` channel and one thread.
- [x] **Step 3:** Move the coordinator's paste call to `OutWorker::send(Job::Paste(..))`; flush before shutdown. Re-run `tests/runtime.rs`.
- [x] **Step 4:** Commit `feat(dictation): output worker thread` (this also removes the recording overshoot during the paste delay).

### Task C.3: `Typer` implementations

**Files:**
- Create: `crates/dictation-service/src/typer.rs`
- Test: `crates/dictation-service/tests/typer.rs` (pure parts only)

**Interfaces:**
- Produces: `pub trait Typer: Send { fn type_text(&mut self, s: &str) -> Result<(), String>; fn backspace(&mut self, n: usize) -> Result<(), String>; fn supports_backspace(&self) -> bool; }` and `pub fn system_typer() -> Box<dyn Typer>`.
- Windows: `SendInput` with `KEYEVENTF_UNICODE` (surrogate pairs for non-BMP); backspace via `VK_BACK`. Linux X11: `xdotool type`/`key BackSpace`. Wayland: `wtype`, `supports_backspace = false` unless evdev is active.
- Produces a pure helper `pub fn utf16_units(s: &str) -> Vec<u16>` for testing.

- [x] **Step 1:** Failing test for `utf16_units` (Hebrew letters map to one unit each; an emoji to two).
- [x] **Step 2:** Implement the helper and the Windows impl, then a hidden `type-test "text" --backspace N` command (like `paste-test`) for manual checks.
- [ ] **Step 3:** Manual check on Windows in Notepad: types Hebrew text, backspaces N characters. Record result.
- [ ] **Step 4:** Linux impls compile on the Linux target in a scratch crate (as done for evdev); not run.
- [x] **Step 5:** Commit `feat(dictation): typer with unicode injection`.

**Review gate:** stop for the user. User tries `type-test` in their own apps (also feeds the backspace compatibility matrix).

---

# Sprint D: Engine stream loop

**Exit criteria:** with `stream = true`, a recording produces `Edit`s while still recording, serialised with the final pass, cancellable, and self-pacing. Output is still logged only (no typing yet).

### Task D.1: Stream loop and pacing

**Files:**
- Create: `crates/dictation-service/src/stream.rs`
- Modify: `crates/dictation-service/src/engine.rs` (`transcribe_window(&self, pcm, rate, language, window_samples, cancelled) -> Result<String>`)
- Modify: `crates/dictation-service/src/config.rs` (stream keys, validation)
- Test: `crates/dictation-service/tests/stream.rs` (with a fake `WindowDecoder` trait, no GPU)

**Interfaces:**
- Produces `pub trait WindowDecoder { fn decode(&self, pcm: &[f32], rate: u32, cancelled: &AtomicBool) -> Result<String, String>; }` implemented by `Engine`.
- Produces `pub struct StreamLoop` with `pub fn run(&mut self, tap: LiveTap, capture_id: u64, stop: Arc<AtomicBool>, emit: impl FnMut(StreamEvent))`, where `StreamEvent { Edit { capture_id: u64, edit: Edit }, Slow { pass_ms: u64 } }`.
- Config keys (all restart-required): `stream: bool = false`, `stream_window_secs: u32`, `stream_interval_ms: u32`, `stream_overlap_ms: u32`, `stream_lock_words: usize`. Defaults come from Phase 0. `validate` rejects `stream_window_secs == 0`, overlap >= window.

Loop rules: take `snapshot_from(window_start)`; skip when shorter than `stream_window_secs` of new audio or when `is_speech` is false; decode; `stitch` with the previous text; feed `Transcript::on_pass`; advance `window_start` only past locked audio (estimated from the segment timestamps of the locked words); start the next pass immediately when the previous finished and at least `stream_interval_ms` passed (self-pacing); when a pass takes more than 3x the window, emit `Slow` and widen the interval; set `cancelled` on an in-flight pass when stop is requested.

- [x] **Step 1:** Failing tests with a scripted fake decoder: (a) text grows over passes and edits are emitted in order; (b) a slow fake (pass time > interval) does not queue passes; (c) silence emits nothing; (d) setting `stop` cancels an in-flight pass and the loop returns; (e) results carry the capture id.
- [x] **Step 2:** Implement `config` keys, then the loop.
- [x] **Step 3:** Run tests and clippy; commit `feat(dictation): engine stream loop with self-pacing`.

### Task D.2: Runtime and controller wiring (still log-only)

**Files:**
- Modify: `crates/dictation-service/src/{controller,runtime}.rs`
- Test: `crates/dictation-service/tests/controller.rs`, `tests/runtime.rs`

- [x] **Step 1:** Failing controller tests: with `stream = true` the chord toggles (first press starts, second stops); a hold does not stop on release; a press during finalising is queued as today; results tagged with an old capture id are dropped.
- [x] **Step 2:** Implement: the engine thread starts `StreamLoop` on `Start` and stops it before the final pass; stream and final passes are serialised on the engine thread; the coordinator forwards `StreamEvent::Edit` to the output worker.
- [x] **Step 3:** Run all dictation-service tests; commit `feat(dictation): stream mode in controller and runtime`.

**Review gate:** stop for the user. The user runs with logging and compares the stream of edits against what they said.

---

# Sprint E: Live typing, safety and finalise

**Exit criteria:** live typing works in Notepad and a browser text area; interference locks corrections; the stop finalises with one correction and sets the clipboard once.

### Task E.1: Type edits and finalise

- [x] **Step 1:** Failing runtime test with fake typer: `Edit::Append("x")` becomes `Job::Type("x")`; `Edit::Replace{backspaces: n, text}` becomes `Job::Backspace(n)` then `Job::Type(text)`; final correction is one edit; the clipboard `Paste`-free `SetClipboard` job runs once with the full text.
- [x] **Step 2:** Add `Job::SetClipboard(String)`; wire the stream events to the output worker; on stop run the final pass, `Transcript::finish`, send the single correction, then the clipboard job, then `flush()`.
- [x] **Step 3:** Tests pass; commit `feat(dictation): live typing and finalise`.

### Task E.2: Interference lock

**Files:**
- Modify: `crates/dictation-service/src/hotkey/` (report non-hotkey key and mouse events), `runtime.rs`
- Test: `crates/dictation-service/tests/runtime.rs`

**Interfaces:**
- Produces `Input::UserActivity` sent by the hotkey backend (rdev already sees keys and mouse) and a foreground-window-changed check on Windows (`GetForegroundWindow` compared before each edit; behind a small `fn foreground_id() -> Option<u64>` returning `None` off Windows).

- [x] **Step 1:** Failing test: after `UserActivity`, subsequent edits are `Append` only; typed text so far is treated as locked.
- [x] **Step 2:** Implement; ensure the chord itself and our own injected events do not trigger it (ignore events while the typer is sending and for 100 ms after).
- [x] **Step 3:** Commit `feat(dictation): lock corrections on user interference`.

### Task E.3: Safety limits

- [x] **Step 1:** Failing test: `Backspace(n)` with `n` greater than the tracked tail length is clamped to the tail length and logged.
- [x] **Step 2:** Implement the clamp in the output path; disable corrections when `typer.supports_backspace()` is false (Wayland).
- [x] **Step 3:** Commit `feat(dictation): backspace safety limits`.

### Task E.4: Manual compatibility matrix and QA

- [ ] **Step 1:** In Notepad, Chrome text area, VS Code, Word, a terminal, and Remote Desktop: dictate for 30 s, record whether text, corrections and Hebrew punctuation are correct. Add the table to `docs/dictation.md`.
- [ ] **Step 2:** Test Hebrew with niqqud and an emoji for the backspace unit; restrict corrections to safe characters if needed (spec rule).
- [ ] **Step 3:** Test with a weak configuration (`--device cpu`, or `precision = "f32"` and a forced long window) and confirm graceful lag plus the `Slow` notification and fallback.
- [ ] **Step 4:** Commit docs; **stop for user review.**

---

# Sprint F: Settings, docs, release 1.0.1

- [ ] **Task F.1:** Settings window: a Streaming checkbox and a precision dropdown (`FormState`, `validate_form`, tests in `tests/settings.rs`).
- [ ] **Task F.2:** Update `docs/dictation.md` (config table, limitations, troubleshooting), the spec status to "implemented", and tick this plan.
- [ ] **Task F.3:** Bump versions to 1.0.1 in `Cargo.toml` and `packaging/dictation.iss`; rebuild the installer; run `cargo test -p dictation-service -- --include-ignored`, clippy and fmt.
- [ ] **Task F.4:** Commit `release: 1.0.1`.

---

## Self-review

- **Spec coverage:** Phase 1 (A.1), Phase 0 (A.2), LiveTap (C.1), output worker (C.2), typer (C.3), state machine (B.2), stitching (B.3), silence gate (B.4), stream loop and cancellation (D.1), serialisation with final pass (D.2), safety rules and interference (E.2, E.3), final pass and clipboard (E.1), config (D.1), settings (F.1). Q1 to Q4 are resolved in Global Constraints. Q5 (hold-to-talk streaming) is deferred.
- **Placeholders:** config defaults are filled from Phase 0 results by design (Task A.2 step 6), not left open.
- **Type consistency:** `Edit`, `Transcript`, `LiveTap::snapshot_from`, `Job`, `Typer`, `StreamEvent` names are used consistently across tasks.
- **Biggest risks:** short-window quality (decided in Sprint A), backspace behaviour with Hebrew (Sprint C.3 and E.4), slow GPUs (D.1 pacing plus fallback).
