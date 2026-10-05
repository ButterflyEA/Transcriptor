//! Types text and backspaces into the focused window, for streaming dictation.
use std::process::Command;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};

use crate::output::is_wayland;

pub trait Typer: Send {
    fn type_text(&mut self, s: &str) -> Result<(), String>;
    fn backspace(&mut self, n: usize) -> Result<(), String>;
    /// False where corrections cannot be made safely (append-only mode).
    fn supports_backspace(&self) -> bool;
}

/// A typer that refuses to type; used where streaming is not enabled.
pub struct NullTyper;

impl Typer for NullTyper {
    fn type_text(&mut self, _: &str) -> Result<(), String> {
        Err("typing is not enabled".into())
    }
    fn backspace(&mut self, _: usize) -> Result<(), String> {
        Err("typing is not enabled".into())
    }
    fn supports_backspace(&self) -> bool {
        false
    }
}

pub fn system_typer() -> Box<dyn Typer> {
    if is_wayland() {
        Box::new(WtypeTyper)
    } else {
        Box::new(EnigoTyper)
    }
}

/// Windows (SendInput with Unicode) and X11, through enigo.
struct EnigoTyper;

impl EnigoTyper {
    fn enigo() -> Result<Enigo, String> {
        Enigo::new(&Settings::default()).map_err(|e| format!("keyboard unavailable: {e}"))
    }
}

impl Typer for EnigoTyper {
    fn type_text(&mut self, s: &str) -> Result<(), String> {
        Self::enigo()?
            .text(s)
            .map_err(|e| format!("typing failed: {e}"))
    }

    fn backspace(&mut self, n: usize) -> Result<(), String> {
        let mut e = Self::enigo()?;
        for _ in 0..n {
            e.key(Key::Backspace, Direction::Click)
                .map_err(|e| format!("backspace failed: {e}"))?;
        }
        Ok(())
    }

    fn supports_backspace(&self) -> bool {
        true
    }
}

/// Wayland: `wtype` types text, but corrections are off until evdev is active.
struct WtypeTyper;

impl Typer for WtypeTyper {
    fn type_text(&mut self, s: &str) -> Result<(), String> {
        let status = Command::new("wtype")
            .arg("--")
            .arg(s)
            .status()
            .map_err(|e| format!("wtype: {e}"))?;
        status
            .success()
            .then_some(())
            .ok_or(format!("wtype exited with {status}"))
    }

    fn backspace(&mut self, _: usize) -> Result<(), String> {
        Err("backspace is not supported on Wayland".into())
    }

    fn supports_backspace(&self) -> bool {
        false
    }
}
