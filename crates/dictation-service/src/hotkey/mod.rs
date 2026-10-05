use crate::chord::{Chord, HotkeyEvent};
use crate::output::is_wayland;

pub mod rdev_backend;

#[cfg(all(target_os = "linux", feature = "evdev"))]
pub mod evdev_backend;

pub trait HotkeyBackend: Send {
    fn name(&self) -> &'static str;
    /// Starts listening on a background thread and returns immediately.
    fn spawn(
        self: Box<Self>,
        chord: Chord,
        on_event: Box<dyn Fn(HotkeyEvent) + Send>,
    ) -> Result<(), String>;
}

/// Picks the global-hotkey backend for this session, or `None` when the
/// control CLI (`dictation start/stop`) is the only way to trigger it.
pub fn select() -> Option<Box<dyn HotkeyBackend>> {
    if cfg!(target_os = "linux") && is_wayland() {
        // Wayland apps cannot grab global keys. Use evdev if compiled in;
        // otherwise compositor binds call the CLI.
        #[cfg(all(target_os = "linux", feature = "evdev"))]
        return Some(Box::new(evdev_backend::EvdevBackend));
        #[cfg(not(all(target_os = "linux", feature = "evdev")))]
        return None;
    }
    Some(Box::new(rdev_backend::RdevBackend))
}
