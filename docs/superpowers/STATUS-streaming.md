# Streaming dictation (v1.0.1): status handoff

Written 2026-10-05 for a fresh session. Read this first.

## Project
- Repo `ButterflyEA/Transcriptor`, `d:\Transcriptor`, branch `feature/dictation-service`.
- `crates/whisper-burn` is the user's Whisper crate. `crates/dictation-service` is a background push-to-talk dictation app. It is fully offline and uses the ivrit-ai Hebrew `whisper-large-v3-turbo` model on GPU through burn/wgpu with f16.
- v1.0.0 is shipped, with an installer in `packaging/dictation.iss`. f16 is the default.
- Plan: `docs/superpowers/plans/2026-10-05-streaming-dictation.md`. Spec: `docs/superpowers/specs/2026-10-05-streaming-dictation-design.md`.
- User rule: do one sprint, validate it, stop and report, and only continue after the user says so. The user is Hebrew-speaking and tests on Windows 11 with an RTX 5060 Ti.

## Streaming goal
With `stream = true` in `%APPDATA%\dictation\config.toml`, the hotkey (Ctrl+Space) is tap-to-toggle. While you speak, text is typed live into the focused window. On stop, the text is corrected once to the full-recording transcript and put on the clipboard.

## Sprint status
- **A (short windows), B (transcript logic), C (tap, output worker, typer) and D (stream loop):** done, committed, tested.
- **E (live typing, interference lock, safety):** code is written and committed (`a914f7e`, `489469b`, `4310f17`). Unit and real-model tests pass with a fake typer. **Real typing into Windows apps is BROKEN**, so E.4 (the manual compatibility matrix) is not done.
- **F (settings checkbox, docs, version 1.0.1, installer):** not started.

## The open bug: garbage typed into Notepad
- **Symptom:** when live typing into Notepad, the user sees things like `שלום ....... הה ssssssssss` or `שלום cccccccccccc`. Letters repeat endlessly.
- **What we know:**
  - The debug log shows the edits and typed text we computed are correct. Log is `%TEMP%\dictation-debug.log`, enabled with the `DICTATION_DEBUG=1` environment variable.
  - It happens even with `dictation type-test "שלום עולם..." --backspace 3` run with no service and no hotkey hook, so `rdev` is NOT the cause.
  - It happened with both the old `enigo` typer and my new `WinTyper` in `src/typer.rs`, which uses raw `SendInput` with `KEYEVENTF_UNICODE`.
  - **The earliest manual `type-test` (Sprint C, `enigo`) "worked well" for the user.** Check what differs: run conditions, console, text.
  - It looks like a stuck, auto-repeating key. Typed characters are held down: `ccccc`, `sssss`, `ההה`.
- **Hypotheses:**
  - (1) A key-up is missing or dropped when the batch is large. The Unicode scan-code key-up must carry the same scan code. Check `send_keys` chunking and `KEYBDINPUT`.
  - (2) The physical Ctrl or Space keys are still held when typing starts, from the Ctrl+Space press that started the recording. The `type-test` run does not have this problem, so this does not explain the garbage there.
  - (3) Terminal and IME interaction.
  - (4) A `windows-sys` struct layout or flag issue. `KEYEVENTF_UNICODE` with `wVk = 0` is the standard form.
- **Experiments built but not yet run by the user** (env var `DICTATION_TYPE_MODE`, read in `WinTyper::type_text`):
  - `paced`: one character at a time, 8 ms gaps.
  - `paste`: clipboard plus Ctrl+V per chunk.
  - Run: `$env:DICTATION_TYPE_MODE="paced"; d:\Transcriptor\target\release\dictation.exe type-test "שלום עולם, בדיקה 123 abc" --backspace 3`, click Notepad within 3 s, then repeat with `paste`.
- **Fallback if typing stays flaky:** use clipboard plus Ctrl+V for each edit, with save and restore of the clipboard. Or drop live typing to a simpler design.
- **An automated reproduction attempt** (WinForms TextBox plus `type-test` in a hidden process) returned empty because the focus is not controllable from a script. Try with the user watching Notepad.

## Other known issues and findings
- **Noise hallucination:** in silence or noise the model invents text ("We have this", "You are"). The RMS gate (`vad.rs`, `DEFAULT_RMS = 0.01`) is too weak. Planned fixes, which the user postponed:
  - wait for about 1.5 s of audio before the first pass;
  - hold back trailing `...` and `?` until the words lock;
  - never emit a pure deletion;
  - strip bidi marks (done, in `stream.rs`);
  - add a repetition guard and a stronger VAD.
- **Stopping:** in the hotkey-off test the user could not stop recording with a second `dictation toggle`. This is unexplained. Check `control.rs` and `runtime.rs` toggle handling in stream mode.
- **Flicker:** edits often append, then backspace the trailing `?` or `.`, then retype.
- **Interference lock:** `activity.rs` is wired to the rdev hook (key presses and mouse clicks). After a lock, the log shows edits without typing. `activity::debug` writes the log lines.
- **Performance:** a padded 3 s window pass takes about 0.9 s. The window is 3 s, the interval is 1 s and the overlap is 400 ms.

## Code map
- `crates/dictation-service/src/`:
  - `stream.rs`: `StreamLoop`, `WindowDecoder`, `StreamEvent`, bidi strip.
  - `transcript.rs`: `Transcript::on_pass` and `finish` (word locking and edits).
  - `stitch.rs`, `vad.rs`, `audio.rs` (`LiveTap`).
  - `outworker.rs`: `Job {Paste, Type, Backspace, SetClipboard, Flush}`.
  - `typer.rs`: `Typer`, `NullTyper`, `WinTyper` (Windows, new), `EnigoTyper` and `WtypeTyper` (non-Windows).
  - `activity.rs`: interference tracking, `plan_sync`, `debug`.
  - `runtime.rs`: coordinator, engine thread, `LiveText` (`apply` and `finish`).
  - `hotkey/rdev_backend.rs`: sends `activity::note_user_input`.
  - `main.rs`: hidden commands `type-test`, `paste-test`, `stream-sweep`, `record-test`. Env `DICTATION_NO_HOTKEY` disables the hotkey.
- Tests are in `crates/dictation-service/tests/`. The real-model runtime tests need `$env:DICTATION_MODEL_DIR="C:\Users\avied\.cache\dictation-models\ivrit-turbo"` and `-- --include-ignored` with `--release`.
- `src/typer.rs`, `src/main.rs`, `Cargo.toml` and `Cargo.lock` have **uncommitted** experimental changes (`DICTATION_TYPE_MODE`, `DICTATION_NO_HOTKEY`). `scripts/sweep.ps1` is untracked, and is a deliverable.
- **Leave alone** (the user's untracked files): `test.txt`, `.vscode/`, the `.wav` files and `results.txt` in the repo root.

## Commits
- `4310f17`: WinTyper, bidi strip, debug log.
- `489469b`: interference lock.
- `a914f7e`: live typing and finalise.
- `8e4fc23`: stream mode runtime.
- `693a9d8`: stream loop.
- `ec5ac3c`: tap, output worker, typer.

## Environment gotchas
- Windows PowerShell 5: no `&&`. Stop processes by PID only.
- A running `dictation.exe` locks `target\release\dictation.exe`, so the build fails with "Access is denied". Stop it by PID first.
- Files use CRLF. For edits, normalise to `\n` and restore. Helper names like `Rd`, `Sv` and `R` collide with aliases. Use `WrX`, `Rep` and so on.
- `cargo test --workspace` hangs. Use `-p dictation-service --test <name>`.
- Do not run `cargo fmt` on whisper-burn. `cargo fmt -p dictation-service` is fine.
- `target\debug` can grow to 100+ GB. Run `cargo clean --profile dev` if the disk fills.
- Only chat messages reach me. The agent cannot see the user's terminal, so log to a file.
- Commit trailer: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Suggested next steps
1. Get the user's results for the `paced` and `paste` `type-test` modes. Fix the typer (see hypotheses above). Verify with a real Notepad, not only fakes.
2. Investigate why the second `toggle` does not stop recording in stream mode.
3. Apply the noise and flicker fixes: warm-up delay, tail hold-back, no pure deletions, a stronger gate, a repetition guard. Re-test.
4. Finish E.4 (matrix: Notepad, Chrome, VS Code, Word, terminal, RDP), then Sprint F (settings checkbox, docs, bump to 1.0.1 in `Cargo.toml` and `packaging/dictation.iss`, installer rebuild, release commit).
