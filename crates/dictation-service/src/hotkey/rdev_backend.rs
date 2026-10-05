use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use rdev::{EventType, Key};

use super::HotkeyBackend;
use crate::chord::{Chord, ChordTracker, HotkeyEvent};

pub struct RdevBackend;

/// Normalized key name used in chords: left/right modifiers collapse to one
/// name, letters and digits are bare (`a`, `1`), function keys are `f1`...
pub fn key_name(key: Key) -> Option<String> {
    let name = match key {
        Key::ControlLeft | Key::ControlRight => "ctrl",
        Key::ShiftLeft | Key::ShiftRight => "shift",
        Key::Alt | Key::AltGr => "alt",
        Key::MetaLeft | Key::MetaRight => "meta",
        Key::Space => "space",
        Key::Return => "enter",
        Key::Escape => "esc",
        Key::Tab => "tab",
        Key::Backspace => "backspace",
        Key::CapsLock => "capslock",
        Key::Unknown(_) => return None,
        other => {
            let dbg = format!("{other:?}");
            let trimmed = dbg
                .strip_prefix("Key")
                .or_else(|| dbg.strip_prefix("Num"))
                .unwrap_or(&dbg);
            return Some(trimmed.to_ascii_lowercase());
        }
    };
    Some(name.to_string())
}

impl HotkeyBackend for RdevBackend {
    fn name(&self) -> &'static str {
        "rdev"
    }

    fn spawn(
        self: Box<Self>,
        chord: Chord,
        on_event: Box<dyn Fn(HotkeyEvent) + Send>,
    ) -> Result<(), String> {
        let (err_tx, err_rx) = mpsc::channel::<String>();
        thread::Builder::new()
            .name("hotkey-rdev".into())
            .spawn(move || {
                let mut tracker = ChordTracker::new(chord);
                let result = rdev::listen(move |event| {
                    let (key, down) = match event.event_type {
                        EventType::KeyPress(k) => (k, true),
                        EventType::KeyRelease(k) => (k, false),
                        _ => return,
                    };
                    if let Some(name) = key_name(key)
                        && let Some(ev) = tracker.feed(&name, down)
                    {
                        on_event(ev);
                    }
                });
                if let Err(e) = result {
                    let _ = err_tx.send(format!("hotkey listener failed: {e:?}"));
                }
            })
            .map_err(|e| e.to_string())?;
        // `listen` only returns on failure, so an early error is a startup error.
        match err_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(e) => Err(e),
            Err(_) => Ok(()),
        }
    }
}
