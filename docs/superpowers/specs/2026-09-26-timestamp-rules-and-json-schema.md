# Reference JSON Schema + Timestamp Rules - Design

- **Date:** 2026-09-26
- **Status:** Draft for review (brainstorming; approach approved 2026-09-26, Phase 0 spike complete)
- **Path classification:** Architectural (changes a public output format and decoder behavior)

## 1. Goal

Make `whisper-burn` produce reference-faithful segment boundaries and emit
`openai/whisper`'s JSON schema, so the golden suite that already exists in this
repo can actually prove parity with the reference implementation. This unblocks
tagging `1.0.0`.

Success = `golden_encoder` and `golden_large_v3` run un-ignored and pass, and
`--output json` produces a document the reference implementation would also
produce for the same audio.

## 2. Why now: the blocking defects

A Phase 0 spike (running the two heavy goldens with `--ignored`, no code
committed) produced these empirical findings. They are the reason `1.0.0` is
blocked.

| Finding | Detail |
| --- | --- |
| `golden_encoder` **passes** | 24.7 s on ndarray/CPU. Its `#[ignore = "network (downloads tiny weights) + golden files"]` is false: weights are cached in `~/.cache/whisper`, fixtures are committed. |
| `golden_large_v3` **fails** | 35 s on wgpu. Three independent defects, none of them the documented "network + heavy" reason. |
| JSON schema mismatch | We emit `[{start,end,text}]` with numeric seconds. Reference emits `{language, segments:[{start:"00:00.000",...}]}`. The comment at `crates/whisper-burn/src/format.rs:133` claims "the shape and units match"; that claim is false. |
| `language` dropped | The detected language is computed but never returned. See 5.3 - this needs a public API change, not just a parameter. |
| Test reads the wrong channel | `golden_large_v3.rs:22` parses JSON from `out.stdout`, but `--output json` writes a sidecar `<audio>.json` and puts human text on stdout. It can never pass as written. |
| Degenerate segments | 22.1 s of audio yields one segment `0.0 -> 30.0` (Whisper's padded window). Reference yields 3 segments ending at `18.640`. |

Supporting change already made: `golden_large_v3` now passes `--device wgpu`
instead of the hardcoded `cpu`. This is the CLI default, runs on the 16 GB
RTX 5060 Ti, and cut the test from a potential hour to 35 s.

## 3. Decisions (from brainstorming Q&A)

1. **Match the reference JSON schema.** Change our output rather than regenerate
   goldens in our own format. Goldens that mirror our own deviation prove
   nothing about reference parity.
2. **Fix segment boundaries in this release**, not defer.
3. **Tolerances stay** at the documented log-mel `1e-2` and encoder `3e-3`. The
   `tasks.md` targets of `1e-4` / `1e-3` are explicitly out of scope; the
   existing comment attributing the gap to DFT-vs-FFT summation order stands.
4. **No new reference fixtures.** `scripts/cement_reference.py`, `jfk.flac` and
   `expected_segments.txt` are out of scope. The existing fixtures in
   `tests/fixtures/golden_tone.json` and `docs/golden/*.json` are sufficient.
5. **No Python work.** Reference generation is unnecessary; `openai-whisper`
   cannot run on this machine's Python 3.14 anyway.
6. **No CI, no code signing, no Linux builds.** The user builds Linux on
   Omarchy and creates the GitHub release in the GUI.
7. **`transcribe()` returns a result struct.** Option (A) in 5.3: the signature
   changes to return `TranscriptionResult { language, segments }`. One honest
   signature beats two, and `whisper-burn` is `0.1.0` and unreleased, so this
   is the only moment the break is free.
8. **Minimal JSON schema for 1.0.0**: `language` + `segments` with string
   timestamps. `task`, `duration`, top-level `text`, and per-segment
   `id`/`seek`/`tokens` are deferred; adding them later is non-breaking.

## 4. Root cause of the degenerate segments

`crates/whisper-burn/src/transcribe.rs:19-21` states it plainly:

> Decoders run without the reference's timestamp rules, so the model may emit
> `<|notimestamps|>`; such runs have no timestamp boundaries and are assembled
> as one whole-chunk segment (start = seek, end = seek + 30 s).

`crates/whisper-burn/src/decoding/greedy.rs:16-17` anticipates the fix:

> `<|0.00|>`'s base token; timestamp ids sit at `timestamp_begin..`. We do not
> apply timestamp pairing rules yet (that arrives with task 24).

The downstream half already exists: `decoding/segments.rs` implements
reference-format timestamp-bounded assembly (`assemble_segments`,
`timestamp_token_to_ms`). The tokenizer already exposes `timestamp_begin` and
`is_timestamp`. **Only the decoder side is missing.**

## 5. Design

### 5.1 Timestamp rules in the decoders

Reference `whisper/decoding.py` seeds the decode with the first timestamp token
and then alternates two restricted vocabularies:

- At a text position, the model may emit text or a timestamp token.
- Immediately after a text token, the allowed set is **narrowed to timestamp
  tokens only** until one is chosen, which closes the current segment.
- The first sampled token is **forced** to the first timestamp token
  (`<|0.00|>`), so the run always starts on a boundary.

Changes:

- `decoding/greedy.rs`: add a timestamp-rule state machine over the decode
  loop; apply the allowed-token mask to logits **before** argmax.
- `decoding/beam.rs`: apply the same mask **per hypothesis**, so each beam
  candidate tracks its own text/timestamp phase. This is materially more
  involved than the greedy case and is the highest-risk item in this design.
- Shared helper (new `decoding/timestamp_rules.rs`) so greedy and beam cannot
  drift apart.

### 5.2 Segment assembly

- Keep `push_untimestamped_fallback` for genuinely `<|notimestamps|>` runs, but
  it should become a rare path rather than the normal one.
- **Reconcile a documented assumption.** `decoding/segments.rs:9-14` states the
  reference emits timestamps as *adjacent pairs* surrounding each text run.
  With the rules in 5.1 the decoder emits `t0 text t1 text t2`, i.e. one
  boundary per text run, which is not the same shape. This must be verified
  against a real decode and the module doc corrected to describe what the code
  actually does.
- Segment `end` must be clamped to real audio duration, not the padded 30 s
  window, so 22.1 s of audio cannot yield `end = 30.0`.

### 5.3 Reference JSON schema

Target shape, matching the committed goldens:

```json
{
  "language": "he",
  "segments": [
    { "start": "00:00.000", "end": "00:07.800", "text": "..." }
  ]
}
```

- `start` / `end` become `HH:MM:SS.mmm` **strings** (the repo already has
  `srt_time` / `vtt_time` helpers in `format.rs`).
- `format_json` gains a `language: &str` parameter.
- Correct the false doc comment at `format.rs:132-135`, which currently claims
  "the shape and units match".
- Update tests that assert the old shape:
  `whisper-burn/tests/format.rs` and `whisper-burn-cli/tests/output.rs`.

**The API change this requires.** `language` is not currently reachable.
`transcribe()` is declared as:

```rust
pub fn transcribe<B: Backend>(...) -> Result<Vec<TranscriptionSegment>>
```

and `resolved_language` is a *local* variable
(`transcribe.rs:309`), assigned from `detect_language` at
`transcribe.rs:343`, consumed for decoding at `transcribe.rs:346` and
`transcribe.rs:349`, and then **discarded** - `Ok(segments)` at
`transcribe.rs:421` never carries it. `transcribe.rs:77` is the *requested*
language (`None` means auto-detect), not the detected one, so it cannot be
substituted.

Two ways out; **(A) is the decision** (see 3.7):

- **(A) Return a result struct - CHOSEN.** Change the signature to return
  `TranscriptionResult { language: String, segments: Vec<TranscriptionSegment> }`.
  Cleanest long-term, but it breaks every caller: the CLI, the Tauri app, and
  a broad slice of the test suite. Justified because `whisper-burn` is still
  `0.1.0` and unreleased, so this is the cheap moment to break it.
- **(B) Add `transcribe_detailed()` - rejected.** Non-breaking, but leaves two
  APIs where one silently discards information.

`resolved_language` falls back to `"en"` at `transcribe.rs:346`, so the
struct's `language` must carry that same resolved value, not the request.

The committed goldens need only `language` + `segments` (decision 3.8), so
`format_json` takes a `language: &str` and emits:

```rust
json!({
    "language": language,
    "segments": segments.iter().map(|s| json!({
        "start": srt_time(s.start),
        "end": srt_time(s.end),
        "text": s.text,
    })).collect::<Vec<_>>(),
})
```

### 5.4 Test corrections

- `golden_large_v3.rs`: read the sidecar JSON file next to the audio instead of
  parsing stdout.
- Remove the `#[ignore]` from `golden_encoder.rs` (verified passing).
- Remove the `#[ignore]` from `golden_large_v3.rs` once it passes.
- Sweep the remaining 11 ignored tests (`downloads_tiny_checkpoint`,
  `greedy_*`, `beam_*`, `transcribe network::*`, `detects_english_silence`).
  Un-ignore what passes now that weights are cached; keep an ignore only where
  something is genuinely impractical, with a **truthful** reason. No more
  "network" excuses for weights already on disk.
- Update `tasks.md` T27 to the true state, listing what was done and what the
  four deferred items are.

## 6. Risks

1. **DTW is not implemented, so exact boundary equality is unlikely.** The
   reference derives segment boundaries from dynamic time warping over the
   encoder's cross-attention, then merges word timestamps by `segment_size`.
   Correct timestamp rules give the model a real place to put boundaries, but
   the chosen timestamps need not equal the reference's DTW output. Realistic
   goal: non-degenerate boundaries that track speech, with matching text.
   Exact `start`/`end` equality with the goldens is a stretch goal. The
   goldens' `assert_eq!` on timestamp strings may need to become a tolerance.
   **Flagging this as the main scope risk.**
2. **Blast radius beyond the goldens.** Timestamp rules change decoding for
   every transcription. The 131 currently-passing tests, the Tauri app, and
   the existing `docs/golden/*.json` comparisons may all shift. Option (A) in
   5.3 additionally breaks the `transcribe()` signature repo-wide.
3. **Beam search complexity.** Per-hypothesis timestamp phase is the hardest
   part; a bug there may be subtle rather than loud.
4. **Device sensitivity.** Goldens were generated by torch; this run is wgpu.
   Last-digit differences are likely even with correct logic.
5. **Public format break.** Changing `--output json` breaks anyone parsing our
   old array shape. Correct at pre-1.0.0; must be called out in release notes.

## 7. Verification gates

- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` (offline suite)
- `cargo test --workspace -- --ignored` (golden suite, cached weights)
- `npm run build` in `crates/transcriptor/ui`
- `tauri build` from `crates/transcriptor`, then confirm the produced exe
  contains the embedded `assets/index-` marker

## 8. Release gate for 1.0.0

`v1.0.0` is tagged only when `golden_encoder` and `golden_large_v3` are
un-ignored and green, `--output json` matches the reference schema, and
`tasks.md` T27 accurately reflects reality. Otherwise the tag waits again.
