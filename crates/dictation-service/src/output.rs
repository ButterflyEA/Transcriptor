use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};

#[derive(Debug, PartialEq, Eq)]
pub enum PasteOutcome {
    Pasted,
    /// The text is on the clipboard but could not be pasted; the reason says why.
    ClipboardOnly(String),
}

pub trait Paster: Send {
    fn paste(&mut self, text: &str) -> Result<PasteOutcome, String>;
}

pub fn is_wayland() -> bool {
    std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v == "wayland")
        || std::env::var_os("WAYLAND_DISPLAY").is_some()
}

pub fn pick_paste_tool(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    ["wtype", "ydotool"].into_iter().find(|t| has(t))
}

fn tool_exists(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|mut c| {
            let _ = c.wait();
        })
        .is_ok()
}

pub fn system_paster(paste_delay_ms: u64) -> Box<dyn Paster> {
    Box::new(SystemPaster { paste_delay_ms })
}

struct SystemPaster {
    paste_delay_ms: u64,
}

// On Windows, `Key::Unicode` is sent as text input and does not trigger shortcuts.
#[cfg(windows)]
fn v_key() -> Key {
    Key::V
}

#[cfg(not(windows))]
fn v_key() -> Key {
    Key::Unicode('v')
}

impl SystemPaster {
    fn set_clipboard(text: &str) -> Result<(), String> {
        let primary = arboard::Clipboard::new()
            .and_then(|mut c| c.set_text(text.to_owned()))
            .map_err(|e| e.to_string());
        match primary {
            Ok(()) => Ok(()),
            Err(e) if is_wayland() => {
                use std::io::Write;
                let mut child = Command::new("wl-copy")
                    .stdin(Stdio::piped())
                    .spawn()
                    .map_err(|_| format!("clipboard unavailable: {e}"))?;
                child
                    .stdin
                    .take()
                    .ok_or("wl-copy stdin")?
                    .write_all(text.as_bytes())
                    .map_err(|e| e.to_string())?;
                child.wait().map_err(|e| e.to_string())?;
                Ok(())
            }
            Err(e) => Err(format!("clipboard unavailable: {e}")),
        }
    }

    fn synthesize_wayland() -> Result<PasteOutcome, String> {
        let Some(tool) = pick_paste_tool(tool_exists) else {
            return Ok(PasteOutcome::ClipboardOnly(
                "install wtype or ydotool to paste automatically".into(),
            ));
        };
        let status = if tool == "wtype" {
            Command::new("wtype")
                .args(["-M", "ctrl", "v", "-m", "ctrl"])
                .status()
        } else {
            Command::new("ydotool")
                .args(["key", "29:1", "47:1", "47:0", "29:0"])
                .status()
        };
        match status {
            Ok(s) if s.success() => Ok(PasteOutcome::Pasted),
            Ok(s) => Ok(PasteOutcome::ClipboardOnly(format!(
                "{tool} exited with {s}"
            ))),
            Err(e) => Ok(PasteOutcome::ClipboardOnly(format!("{tool}: {e}"))),
        }
    }

    fn synthesize_enigo() -> Result<PasteOutcome, String> {
        let mut enigo = match Enigo::new(&Settings::default()) {
            Ok(e) => e,
            Err(e) => {
                return Ok(PasteOutcome::ClipboardOnly(format!(
                    "keyboard unavailable: {e}"
                )));
            }
        };
        let result = enigo.key(Key::Control, Direction::Press).and_then(|()| {
            let click = enigo.key(v_key(), Direction::Click);
            // Always release Ctrl, even if the click failed.
            enigo.key(Key::Control, Direction::Release)?;
            click
        });
        Ok(match result {
            Ok(()) => PasteOutcome::Pasted,
            Err(e) => PasteOutcome::ClipboardOnly(format!("could not send Ctrl+V: {e}")),
        })
    }
}
impl Paster for SystemPaster {
    fn paste(&mut self, text: &str) -> Result<PasteOutcome, String> {
        Self::set_clipboard(text)?;
        sleep(Duration::from_millis(self.paste_delay_ms));
        if is_wayland() {
            Self::synthesize_wayland()
        } else {
            Self::synthesize_enigo()
        }
    }
}
