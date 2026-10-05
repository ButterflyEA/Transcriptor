# Streaming dictation (v1.0.1) — design spec

Status: draft for review. Nothing here is implemented. Builds on
[2026-10-04-dictation-service-design.md](./2026-10-04-dictation-service-design.md).

## Goal

While the user is dictating, text appears in the focused field within about a
second of being spoken, and the last few words are corrected as later audio
arrives. When dictation stops, the remaining tail is finalised quickly instead
of the current wait of several seconds.

Everything stays offline, on the bundled ivrit-ai Hebrew model, within the
4 GB VRAM target.

## Decisions (from review)

| # | Decision |
|---|---|
| D1 | Main goal: live text typed into the document while the user is still talking. |
| D2 | Streaming mode is **tap to start, tap to stop** (same chord). No modifier is held while text is injected. Hold-to-talk with batch decoding stays as the default mode. |
| D3 | Text typed live may be **corrected with backspaces**, but only inside a bounded *tentative tail*. Older text is *locked* and never changed. |
| D4 | Target all platforms. Corrections are best-effort off Windows. Where corrections cannot be made safe (Wayland without input observation), fall back to append-only. |
| D5 | Configuration is restart-required, matching the existing convention. |

## Architecture

### Phase 1 — whisper-burn: short windows

1. `Encoder::forward` slices the positional embedding to the real sequence
   length after the conv stack. At 3000 frames the slice is the whole table, so
   existing golden tests stay byte-identical.
2. `TranscriptionOptions.window_samples: Option<usize>`. `None` keeps the
   current 30 s path exactly. Opt-in only.
3. A sibling of `split_chunks` that returns start offsets. Remove the hardcoded
   `seek_ms = ci * 30_000` and the hardcoded `N_FRAMES` in the tensor shape;
   both derive from the actual window. Keep `split_chunks` for the 30 s path.
4. Latent bugs to fix on the way:
   - `stft.rs` reflect padding panics on buffers under about 202 samples.
     Return a clear error.
   - `encoder.rs` `half = out_len / 2` drops the tail when `out_len` is odd.
     Constrain streaming windows to even frame counts (multiples of 20 ms).
5. Check the log-mel normalisation: Whisper clamps to the window's own maximum,
   so short windows can change the features pass to pass and make text flicker.
   Measure it in Phase 0 and normalise with a fixed reference if needed.

### Phase 2 — dictation-service plumbing

6. **`LiveTap`** in `audio.rs`: a cloneable handle to the in-progress capture
   (buffer, rate, max). `snapshot_from(offset)` copies only the samples after
   `offset`, so a pass never clones or re-resamples the whole buffer.
   `CpalRecorder` publishes the tap on start and clears it on stop.
7. **The engine thread reads the tap directly**, not through the coordinator,
   so the decode loop is not in the path that has the stall today.
   - Every stream result carries the capture number. The coordinator drops
     results from an old capture.
   - Stream passes and the final pass are serialised on the engine thread (one
     GPU, one model). A newer pass sets `TranscriptionOptions.cancelled` on an
     in-flight one so stale work aborts instead of queueing. Cancellation is
     between tokens; an encoder pass already running cannot be interrupted.
8. **Output worker thread** owns the `Paster`/typer and drains an `mpsc`
   channel. The coordinator enqueues and returns. A single worker keeps
   ordering. A `Flush(Sender)` job orders shutdown after the final commit. This
   also removes today's recording overshoot caused by pasting on the
   coordinator thread.

### Phase 3 — transcript state: locked text and tentative tail

State per capture: `window_start` (audio offset), `locked` (text already final),
`typed_tail` (text currently typed but still correctable, with its exact
length in the units the target app deletes), `last_tokens`.

Each pass, on new audio:

1. Snapshot from `window_start`. Skip if there is less than one window.
2. Gate with an energy/VAD check so silent windows do not hallucinate text.
3. Decode the window (with overlap stitching, see below) into tokens `T_n`.
   Normalise before comparing: strip punctuation and niqqud differences so
   `,` versus `.` flips do not break agreement.
4. `agreed = longest_common_prefix(T_n, last_tokens)` on **word tokens**, never
   inside a word. Words in `agreed` older than the lock horizon become locked.
5. The tentative tail is `T_n` after the locked text. If it differs from
   `typed_tail`, send the output worker one edit: N backspaces plus the new
   tail text. If it only extends the typed tail, type the extension.
6. Advance `window_start` to the audio offset of the first non-locked word, using
   word timestamps (or segment offsets as a fallback). Locked audio is never
   decoded again.
7. `last_tokens = T_n`.

**Invariant (property-tested):** locked text is a prefix of every later pass's
accepted text. A violation means a pass contradicted locked text; the policy is
to keep what is typed, append only the non-contradicting suffix, and log it.

**Overlap stitching:** overlapping windows duplicate words at joins. Merge by
aligning the overlap on normalised tokens and dropping the duplicated run. This
is its own tested unit.

**Final pass:** on stop, decode the remaining tail once, diff it against
`typed_tail`, and send one correction. Never retype the whole text. Set the
clipboard once, to the full final transcript.

### Safety rules for backspace corrections

Backspace deletes whatever precedes the caret, not necessarily our text.

- Never backspace more characters than the length of our own `typed_tail`, and
  never past the start of this session's typed text.
- On any non-hotkey keystroke, mouse click, or change of the foreground window,
  **lock everything and stop correcting**; streaming continues append-only or
  stops (open question Q4). On Windows the existing hook sees keys, and the
  foreground window handle can be compared before each edit.
- The tail's length is measured in the unit the target deletes. Backspace
  removes a grapheme in some apps and a code point in others (Hebrew with
  niqqud, emoji, RTL marks). Until measured, restrict corrections to plain
  letters, spaces and common punctuation.
- Windows: type deltas with `SendInput` Unicode events, not the clipboard.
  Linux/X11: `enigo`/xdotool-style typing. Wayland: `wtype`/`ydotool`, with
  corrections **disabled by default** until measured (D4 fallback), because the
  user's own input cannot be observed without evdev.
- The clipboard is not touched per delta.

### Known weaknesses

- A self-correction ("X, no, Y") can be locked before the correction arrives.
  Bounded by the lock horizon; visible as a short flicker in the tail.
- A pass slower than the interval stalls the stream. Mitigated by cancellation
  and by measuring pass time per GPU in Phase 0.
- Quality of short windows on the ivrit model is unproven. It was trained on
  30 s padded input, so short windows are a distribution shift.

## Configuration

```toml
stream = false              # opt in
stream_window_secs = 3      # placeholder until the Phase 0 sweep
stream_interval_ms = 1000
stream_overlap_ms = 400
stream_lock_words = 3       # words kept tentative before locking (placeholder)
```

All restart-required. `stream = true` switches the hotkey to tap-to-toggle.

## Phase 0 — measurements before building Phase 3

1. WER on Hebrew versus window length (1, 2, 3, 5, 10 s), against the golden
   reference and a small set of recorded dictation clips.
2. Pass time per window length on the dev GPU and, if available, on a 4 GB card.
3. How often the last 3 s of text changes between consecutive passes.
4. A backspace compatibility matrix: Notepad, a browser text area, VS Code,
   Word, a terminal, a remote desktop session, and Hyprland with `wtype`.
5. Unit of deletion for Hebrew with niqqud in each of those apps.

Go/no-go: proceed only if short-window WER is within the budget in Q3 and pass
time at the chosen window is below the interval on the minimum GPU in Q2.

## Decisions on the open questions

- **Q1 latency:** about 1.5 s is acceptable; a few more word errors are accepted.
- **Q2 weak GPUs:** streaming degrades instead of disabling. Self-pacing (next pass starts when the previous finished), a widened interval, a notification, and a fall back to batch mode for the capture if a pass takes more than 3x the window.
- **Q3 WER budget:** measured in Phase 0 and agreed with the user before building Phase 3.
- **Q4 interference:** on typing, click or window change, lock everything typed so far and continue append-only.
- **Q5 hold-to-talk streaming:** deferred.

## Plan impact

Implemented by [../plans/2026-10-05-streaming-dictation.md](../plans/2026-10-05-streaming-dictation.md) (v1.0.1): Phase 0 spike, then Phases 1–3 as sprints,
with the commit-state module developed test-first and property-tested before
any wiring to the engine or typing.

## Phase 0 results (measured on the RTX 5060 Ti, f16, 22 s Hebrew clip)

| Window | Unpadded CER | Padded-to-30 s CER | Padded ms/pass |
|---|---|---|---|
| 1 s | 13.5 (loops) | 0.236 | 816 |
| 2 s | 7.6 (loops) | 0.133 | 960 |
| 3 s | 4.6 (repeats) | 0.091 | 972 |
| 5 s | 1.25 (repeats) | 0.055 | 1135 |
| 10 s | 0.061 | n/a | n/a |

Findings:
- The model needs 30 s zero-padded input. Short unpadded windows hallucinate, so streaming uses padded windows (`pad_windows = true`).
- The naive CPU mel was the bottleneck (2.1 s). Parallel frames plus zero-frame skip took it to 0.17 s, which also speeds up batch dictation.
- Per-pass time is now about 1 s. The encoder costs about 0.2 s whatever the window length, and decode costs about 55 ms per token.
- Silence-only windows can hallucinate "תודה רבה", so the VAD gate in Sprint B is required.
- Not yet measured: extra recorded clips, tail stability, and mel drift on live audio.

Recommended defaults: 3 s padded windows, interval about 1 s, which gives about 1.5 s latency.

### Recorded-clip results (padded windows, CER vs the full 30 s decode)

| Clip | 1 s | 2 s | 3 s | 5 s | ms/pass (3 s) |
|---|---|---|---|---|---|
| normal (22 s) | 0.280 | 0.125 | 0.060 | 0.036 | 916 |
| pauses (22 s) | 0.564 | 0.359 | 0.154 | 0.128 | 854 |
| silence (22 s) | empty | empty | empty | empty | 699 |
| test clip (22 s) | 0.236 | 0.133 | 0.091 | 0.055 | 907 |

- Silence produced empty text at every window size, so there were no hallucinations on pure silence. The 'תודה רבה' hallucination only appeared on trailing audio, so the VAD gate is still worth keeping.
- Pauses are the weak case. At 3 s the error rate is 0.15, because windows that split a sentence near a pause drift.
- Go: 3 s padded windows are the default (error rate 0.06-0.15, about 0.9 s per pass). 2 s is an option for lower latency, at 0.13-0.36.
