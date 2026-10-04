# Dictation Service — Design

Status: draft for review. No implementation has started.

## Goal

A background service built on the `whisper-burn` crate. The user holds a global
hotkey, speaks, releases the key, and the transcript is pasted into the focused
text field and also left on the clipboard. All inference is local.

## Decisions

| Topic | Decision |
|---|---|
| Platforms | Windows and Linux (X11 and Wayland, incl. Hyprland/Omarchy via compositor binds) |
| Interaction | Push-to-talk: hold to record, release to transcribe |
| Output | Always paste into the focused field AND leave the text on the clipboard |
| UI | Tray icon + small settings window |
| Language | User pins one language in settings; "auto" is an option |
| Stack | Pure Rust, single lean binary (no web stack) |
| Location | New workspace crate `crates/dictation-service` (binary), depends on `whisper-burn` |
| Network | None, ever (airgapped). Installer bundles an ivrit Hebrew model sized for 4 GB VRAM |

## Architecture

Units with one purpose each, communicating over channels:

- **config** — TOML file in the platform config dir. Fields: hotkey, model,
  language (`auto` or code), input device, paste-delay. Load/save/validate only.
- **hotkey** — a trait with several backends, all emitting `Pressed` /
  `Released` events for the configured chord:
  - Windows: low-level keyboard hook.
  - Linux X11: `rdev`.
  - Linux Wayland (e.g. Omarchy/Hyprland): the **control** backend (below) is
    the primary path; an optional **evdev** backend reads `/dev/input` directly
    (requires the user to be in the `input` group; documented privacy tradeoff).
  Rebinding by capture in the settings window applies to the in-process
  backends only; with the control backend the binding lives in the compositor.
- **control** — local IPC (Unix socket on Linux, named pipe on Windows) plus
  CLI subcommands `dictation start` / `dictation stop` / `dictation toggle` /
  `dictation status`. Wayland users bind them in the compositor, e.g. Hyprland:
  `bind = CTRL, SPACE, exec, dictation start` and
  `bindr = CTRL, SPACE, exec, dictation stop`. Also usable for scripting on
  every platform. The CLI is the same binary in client mode.
- **capture** — `cpal` input stream. While the key is held, buffers samples,
  downmixes to mono and reports its native sample rate (`whisper-burn`
  resamples). Returns `Vec<f32>` + rate on release. Caps recording length
  (default 120 s).
- **engine** — worker thread that loads the `whisper-burn` model once at start
  and runs a warm-up pass. Accepts `Vec<f32>` + language, returns text.
  Backend: wgpu with ndarray fallback (as in the CLI). Uses the existing
  `whisper_burn::transcribe`, which already takes in-memory PCM plus a sample
  rate and resamples internally; no `whisper-burn` change is needed.
- **output** — sets the clipboard (`arboard`; Wayland via data-control, or
  `wl-copy` fallback), then simulates Ctrl+V: `enigo` on Windows/X11, `wtype`
  on Wayland (virtual-keyboard protocol), `ydotool` as a fallback (needs
  `uinput` access). The clipboard is deliberately NOT restored, per the chosen
  behaviour. Terminals paste with Ctrl+Shift+V; a per-app override is out of
  scope for v1, so this is a documented limitation.
- **ui** — `tray-icon` with states (idle, recording, transcribing, error),
  menu (Settings, Pause, Quit). The settings window is an `egui` app run as a
  separate `dictation settings` process, because the tray and egui each need
  to own the main-thread event loop; it signals the service with `reload`.
- **app** — state machine tying the above together.

## Data flow

`Pressed` → capture starts, tray = recording →
`Released` → capture stops, tray = transcribing → engine → output →
tray = idle. Recordings shorter than ~300 ms are discarded (accidental taps).
Key presses during transcribing start a new recording; its audio is queued
(FIFO, capped) and transcribed and pasted after the current job.

## Model handling

**Airgapped operation is a hard requirement.** The service never opens a network
connection. The installer ships a Hebrew (ivrit-ai) model; the engine loads it
from a local model directory (`model_dir` in config, default: `models/` next to
the executable, then the platform data dir). If the model is missing the service
reports a clear error and does NOT attempt a download. The `weights` (download)
feature of `whisper-burn` is not used by the service.

**Target hardware: 4 GB VRAM.** The shipped model is
`ivrit-ai/whisper-large-v3-turbo` (808M params, Apache-2.0, single F32
`model.safetensors` ~3.2 GB, same architecture as `LargeV3Turbo`). The
existing `IvritHebrew` variant (ivrit-ai/whisper-large-v3, ~6.2 GB resident)
does not fit. `whisper-burn` computes in f32 (~3.3 GB floor before
activations), which is very tight on 4 GB. A **spike (Sprint 1)** measures real
VRAM for f32 and f16 and decides: f32 as-is, f16 compute in `whisper-burn`, or a
fallback to a smaller model. The default model is fixed only after the spike.
Other models can still be placed in `model_dir` and selected in settings.

## Error handling

- No microphone / device lost: tray error state and a notification; retry on
  next press.
- Model missing: tray error stating the expected path; no download attempt
  (airgapped). Out of VRAM at load: error naming the model and suggesting a
  smaller one.
- Paste failure: text is still on the clipboard; notify the user.
- Hotkey conflict or registration failure: settings window prompts a rebind.
- Empty or silent transcript: do nothing, no paste.

## Platform notes / risks

- **Wayland:** apps cannot grab global keys; the compositor owns them.
  Supported via compositor bindings calling the control CLI (primary) or the
  optional evdev backend. If neither the paste tool (`wtype`/`ydotool`) nor
  injection is available, the service degrades to clipboard-only with a
  notification. The XDG GlobalShortcuts portal is a later addition.
- **Windows elevation:** key injection into elevated windows fails unless the
  service is elevated; documented, not worked around.
- **Always-on cost:** the loaded model dominates idle RAM; model choice is the
  user's lever. The tray offers "Unload model" as a possible later addition.
- **Antivirus:** global keyboard hooks can trigger heuristics; unsigned binary
  already carries this caveat.

## Testing

- Unit: config round-trip/validation; resampling/downmix; state machine
  transitions with fake capture/engine/output.
- Integration: engine transcribes a fixture WAV buffer (reuse golden audio).
- Manual checklist per platform: hotkey, paste into a few apps, clipboard
  contents, device-loss recovery.

## Out of scope for v1

Toggle mode, voice-activity auto-stop, streaming/partial text, macOS, per-app
profiles, clipboard restore, autostart installer (autostart toggle is a
candidate for v1.1).

## Resolved questions

1. Default hotkey: Ctrl+Space (rebindable). Chord release = release of either key.
   Known collisions with IME/language switching and editor autocomplete.
2. Press during transcription: queue it. Recording starts immediately and its
   audio is transcribed after the current job, in order. Outputs are pasted
   in order. Queue depth is capped (default 3) with a tray indication when full.
3. Autostart: deferred to v1.1 (Windows Run key / Linux ~/.config/autostart).

## Open questions

None.
