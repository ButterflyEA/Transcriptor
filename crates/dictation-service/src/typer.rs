//! Types text and backspaces into the focused window, for streaming dictation.
#[cfg(not(windows))]
use std::process::Command;

#[cfg(not(windows))]
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

#[cfg(not(windows))]
use crate::output::is_wayland;

pub trait Typer: Send {
    fn type_text(&mut self, s: &str) -> Result<(), String>;
    fn backspace(&mut self, n: usize) -> Result<(), String>;
    /// False where corrections cannot be made safely (append-only mode).
    fn supports_backspace(&self) -> bool;
    /// False for a typer that cannot type at all.
    fn is_enabled(&self) -> bool {
        true
    }
}

/// A typer that refuses to type; used where streaming is not enabled.
pub struct NullTyper;

impl Typer for NullTyper {
    fn is_enabled(&self) -> bool {
        false
    }
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
    #[cfg(windows)]
    return Box::new(WinTyper);
    #[cfg(not(windows))]
    if is_wayland() {
        Box::new(WtypeTyper)
    } else {
        Box::new(EnigoTyper)
    }
}

/// X11, through enigo.
#[cfg(not(windows))]
struct EnigoTyper;

#[cfg(not(windows))]
impl EnigoTyper {
    fn enigo() -> Result<Enigo, String> {
        Enigo::new(&Settings::default()).map_err(|e| format!("keyboard unavailable: {e}"))
    }
}

#[cfg(not(windows))]
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
#[cfg(not(windows))]
struct WtypeTyper;

#[cfg(not(windows))]
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

/// Windows: one SendInput batch per call, so the characters arrive in order
/// and no key can be left held down.
#[cfg(windows)]
struct WinTyper;

#[cfg(windows)]
fn send_keys(
    inputs: &[windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT],
) -> Result<(), String> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{INPUT, SendInput};
    for chunk in inputs.chunks(64) {
        // SAFETY: chunk is a valid slice of INPUT structs and the size matches.
        let sent = unsafe {
            SendInput(
                chunk.len() as u32,
                chunk.as_ptr(),
                std::mem::size_of::<INPUT>() as i32,
            )
        };
        if sent as usize != chunk.len() {
            return Err("typing was blocked by the target window".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    Ok(())
}

#[cfg(windows)]
fn key_input(
    vk: u16,
    scan: u16,
    flags: u32,
) -> windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT,
    };
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

#[cfg(windows)]
impl Typer for WinTyper {
    fn type_text(&mut self, s: &str) -> Result<(), String> {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{KEYEVENTF_KEYUP, KEYEVENTF_UNICODE};
        match std::env::var("DICTATION_TYPE_MODE").as_deref() {
            Ok("paste") => return paste_text(s),
            Ok("paced") => {
                for unit in s.encode_utf16() {
                    send_keys(&[key_input(0, unit, KEYEVENTF_UNICODE)])?;
                    std::thread::sleep(std::time::Duration::from_millis(8));
                    send_keys(&[key_input(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)])?;
                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
                return Ok(());
            }
            _ => {}
        }
        let mut inputs = Vec::new();
        for unit in s.encode_utf16() {
            inputs.push(key_input(0, unit, KEYEVENTF_UNICODE));
            inputs.push(key_input(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
        }
        send_keys(&inputs)
    }

    fn backspace(&mut self, n: usize) -> Result<(), String> {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{KEYEVENTF_KEYUP, VK_BACK};
        let mut inputs = Vec::new();
        for _ in 0..n {
            inputs.push(key_input(VK_BACK, 0, 0));
            inputs.push(key_input(VK_BACK, 0, KEYEVENTF_KEYUP));
        }
        send_keys(&inputs)
    }

    fn supports_backspace(&self) -> bool {
        true
    }
}

/// Inserts text through the clipboard and Ctrl+V, the path the batch dictation uses.
#[cfg(windows)]
fn paste_text(s: &str) -> Result<(), String> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{KEYEVENTF_KEYUP, VK_CONTROL};
    arboard::Clipboard::new()
        .and_then(|mut c| c.set_text(s.to_owned()))
        .map_err(|e| format!("clipboard unavailable: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(40));
    const VK_V: u16 = 0x56;
    send_keys(&[
        key_input(VK_CONTROL, 0, 0),
        key_input(VK_V, 0, 0),
        key_input(VK_V, 0, KEYEVENTF_KEYUP),
        key_input(VK_CONTROL, 0, KEYEVENTF_KEYUP),
    ])?;
    std::thread::sleep(std::time::Duration::from_millis(60));
    Ok(())
}
