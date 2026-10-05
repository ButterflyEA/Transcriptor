# Dictation service

Push-to-talk dictation for Hebrew (and other languages) that runs fully offline.
Hold a key, speak, release: the text is pasted into the focused field and also
left on the clipboard.

## Install and build

Build: `cargo build --release -p dictation-service` (binary: `target/release/dictation`).
Linux build needs ALSA headers (`libasound2-dev`), `pkg-config`, and for the tray
GTK 3 and libayatana-appindicator.

Release bundles include the Hebrew model and need no network at install or run
time; see [packaging/README.md](../packaging/README.md). On Windows unzip the
bundle and run `.\install.ps1`. The service looks for the model in this order:
`model_dir` from the config, `models/<model>` next to the executable,
`<data dir>/dictation/models/<model>`, then on Linux
`/usr/share/dictation/models/<model>`. Nothing is ever downloaded.

`dictation verify-model [folder]` checks the model files against `SHA256SUMS`,
which is how IT can confirm a transfer into an airgapped network arrived intact.

The bundled model is [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo)
(Apache-2.0, see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)). It is
Hebrew-focused with degraded language detection, so `language` defaults to `he`.

## Configuration

File: Windows `%APPDATA%\dictation\config.toml`, Linux `~/.config/dictation/config.toml`.
All fields are optional.

| Field | Default | Meaning |
|---|---|---|
| `hotkey` | `"ctrl+space"` | chord joined with `+`; restart to apply |
| `model` | `"ivrit-turbo"` | `ivrit-turbo` or a stock size such as `tiny`, `large-v3-turbo` (needs its own folder) |
| `model_dir` | unset | folder with `config.json` and `model.safetensors` for `model` |
| `language` | `"he"` | `auto` or a 2-letter lowercase code |
| `input_device` | unset (system default) | input device name; restart to apply |
| `paste_delay_ms` | `50` | wait between setting the clipboard and pasting; restart to apply |
| `max_record_secs` | `120` | a recording is stopped automatically after this long; restart to apply |
| `queue_cap` | `3` | recordings kept waiting while one is transcribing; extra ones are dropped with a notice |

Recordings under 300 ms and silent ones are discarded without pasting.

## Hotkey

The default hotkey is **Ctrl+Space**, push-to-talk: hold to record, release to
transcribe and paste. Change it with `hotkey` in `config.toml` (for example
`hotkey = "ctrl+shift+space"`); restart the service to apply it. Key names are
lowercase: `ctrl`, `shift`, `alt`, `meta`, `space`, `a`-`z`, `0`-`9`, `f1`-`f12`.
A press during transcription is queued.

- **Windows and Linux/X11:** a global listener (rdev) handles the hotkey. The
  keys are observed, not swallowed, so the focused app still receives Ctrl+Space
  as well. Pick a different chord if that bothers you. Windows does not deliver
  keys typed into elevated (administrator) windows to a normal process, so the
  hotkey does nothing there.
- **Linux/Wayland:** apps cannot listen for global keys, so use the compositor
  binds below. Optionally build with `--features evdev` to read `/dev/input`
  directly; this needs your user in the `input` group and lets the process see
  all keystrokes, so it is off by default.

## Tray and settings

`dictation run` shows a tray icon whose colour is the state (grey loading, green
idle, red recording, orange transcribing, purple error). The menu has the
current status, **Paused** (refuses new recordings), **Settings…** and **Quit**.
Problems such as "copied to clipboard, not pasted" appear as desktop
notifications. If no tray is available the service runs headless and
`dictation start/stop` still work. On Linux the tray needs GTK 3 and
libappindicator (for example `libayatana-appindicator3`).

`dictation settings` opens the settings window: hotkey, model, model folder,
language and input device. Save writes `config.toml` and tells a running
service to reload; a changed hotkey needs a service restart.

## Controlling a running service

`dictation run` starts the service (foreground). It listens on a local socket
(Windows: named pipe `dictation`; Linux: `$XDG_RUNTIME_DIR/dictation.sock`,
user-only permissions). Other `dictation` invocations talk to it:

| Command | Effect |
|---|---|
| `dictation start` | start recording |
| `dictation stop` | stop recording and transcribe |
| `dictation toggle` | start if idle, otherwise stop |
| `dictation status` | prints `loading`, `idle`, `recording`, `transcribing` or `error` |
| `dictation reload` | re-read `config.toml`; the model is reloaded only if `model` or `model_dir` changed |

Each prints a one-line reply and exits non-zero if the service is not running.
Only one instance can run at a time. `reload` applies `language` and the model
settings; changes to `hotkey`, `input_device`, `paste_delay_ms`, `max_record_secs`
and `queue_cap` need a restart.

## Linux / Wayland (Omarchy, Hyprland)

Wayland does not let ordinary apps listen for global keys, so bind the keys in
the compositor and have them call the control commands.

Requirements: `wtype` (or `ydotool`) to paste, and `wl-clipboard` as a clipboard
fallback.

`~/.config/hypr/bindings.conf`:

```
exec-once = dictation run
bind  = CTRL, SPACE, exec, dictation start
bindr = CTRL, SPACE, exec, dictation stop
```

`bind` fires on key press and `bindr` on release, which gives push-to-talk.
For tap-to-start/tap-to-stop use a single `bind = CTRL, SPACE, exec, dictation toggle`.

Caveats:
- Terminals paste with Ctrl+Shift+V, so automatic paste does nothing there;
  the text is still on the clipboard.
- If neither `wtype` nor `ydotool` is installed you get the text on the
  clipboard only, with a notice.
- Ctrl+Space may already be bound (for example by an input method); pick
  another chord if so.

## Known limitations

- Elevated (administrator) windows on Windows do not receive the paste and
  never deliver hotkey events.
- Terminals paste with Ctrl+Shift+V; the text is left on the clipboard.
- The hotkey is observed, not swallowed, so Ctrl+Space also reaches the focused
  app (and may collide with input-method switching).
- Pasting runs on the coordinator thread, so a very large `paste_delay_ms`
  delays handling of other keys.

## Troubleshooting

| Symptom | Check |
|---|---|
| Tray is purple / status `error` | The notification names the folders searched. Fix `model_dir`, then tray **Retry model load**. |
| "model still loading" | The model takes several seconds to load at start. |
| Text only on the clipboard | No focused text field, a terminal, or no `wtype`/`ydotool` on Wayland. |
| Nothing happens on hotkey | Wayland: use compositor binds. Windows: the focused window may be elevated. |
| Mic problems | Pick the device in **Settings**. Unplugging mid-recording gives a notice; the next press reopens the device. |
| Another instance | `dictation is already running`; use `dictation status`. |

## Manual QA checklist

Record results per release. Items marked (unverified) have not been run yet.

- [ ] Hold/release records and pastes in Notepad, a browser and VS Code (Windows)
- [ ] Same on X11 and on Hyprland via binds
- [ ] Clipboard contains the text afterwards
- [ ] Press during transcription queues and pastes in order; the 4th rapid recording drops with a notice
- [ ] Taps under 300 ms do nothing; silence pastes nothing
- [ ] Unplug the mic mid-recording: notification, next press works
- [ ] Dictating Hebrew with `language = "he"`
- [ ] Peak VRAM and idle RAM with the bundled model (spike: about 4.0 GB VRAM peak in f32, see the plan's Task 1.3 decision)
- [ ] Airgap: networking disabled, cold start, dictate Hebrew; firewall log shows no outbound connections from `dictation`
