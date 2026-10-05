# Dictation service

Push-to-talk dictation for Hebrew (and other languages) that runs fully offline.
Hold a key, speak, release: the text is pasted into the focused field and also
left on the clipboard.

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
