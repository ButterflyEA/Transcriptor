//! Reads key events straight from `/dev/input`. This works on Wayland but
//! needs read access to input devices (the `input` group), which also means the
//! process can see every keystroke; see docs/dictation.md.

use std::sync::{Arc, Mutex};
use std::thread;

use evdev::{Device, EventSummary, KeyCode};

use super::HotkeyBackend;
use crate::chord::{Chord, ChordTracker, HotkeyEvent};

pub struct EvdevBackend;

pub fn code_name(code: KeyCode) -> Option<String> {
    let dbg = format!("{code:?}");
    let name = dbg.strip_prefix("KEY_")?.to_ascii_lowercase();
    Some(
        match name.as_str() {
            "leftctrl" | "rightctrl" => "ctrl",
            "leftshift" | "rightshift" => "shift",
            "leftalt" | "rightalt" => "alt",
            "leftmeta" | "rightmeta" => "meta",
            other => other,
        }
        .to_string(),
    )
}

fn is_keyboard(d: &Device) -> bool {
    d.supported_keys()
        .is_some_and(|k| k.contains(KeyCode::KEY_A) && k.contains(KeyCode::KEY_ENTER))
}

impl HotkeyBackend for EvdevBackend {
    fn name(&self) -> &'static str {
        "evdev"
    }

    fn spawn(
        self: Box<Self>,
        chord: Chord,
        on_event: Box<dyn Fn(HotkeyEvent) + Send>,
    ) -> Result<(), String> {
        let keyboards: Vec<Device> = evdev::enumerate()
            .map(|(_, d)| d)
            .filter(is_keyboard)
            .collect();
        if keyboards.is_empty() {
            return Err("no readable keyboard in /dev/input; add your user to the `input` group and log in again".into());
        }
        let tracker = Arc::new(Mutex::new(ChordTracker::new(chord)));
        let on_event: Arc<dyn Fn(HotkeyEvent) + Send + Sync> = Arc::from(on_event_sync(on_event));
        for mut device in keyboards {
            let tracker = Arc::clone(&tracker);
            let on_event = Arc::clone(&on_event);
            thread::Builder::new()
                .name("hotkey-evdev".into())
                .spawn(move || {
                    loop {
                        let Ok(events) = device.fetch_events() else {
                            return; // device unplugged
                        };
                        for ev in events {
                            // value 2 is autorepeat
                            if let EventSummary::Key(_, code, value @ (0 | 1)) = ev.destructure()
                                && let Some(name) = code_name(code)
                                && let Some(out) = tracker.lock().unwrap().feed(&name, value == 1)
                            {
                                on_event(out);
                            }
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// The trait hands us a `Send` closure; we share it across device threads
/// behind a mutex to satisfy `Sync`.
fn on_event_sync(f: Box<dyn Fn(HotkeyEvent) + Send>) -> Box<dyn Fn(HotkeyEvent) + Send + Sync> {
    let f = Mutex::new(f);
    Box::new(move |e| (f.lock().unwrap())(e))
}
