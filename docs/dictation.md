# Dictation service

Push-to-talk dictation for Hebrew (and other languages) that runs fully offline.
Hold a key, speak, release: the text is pasted into the focused field and also
left on the clipboard.

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
