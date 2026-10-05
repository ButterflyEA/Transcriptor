//! Tracks whether the user touched the keyboard or mouse while we type, so
//! live corrections can stop before they delete the wrong text.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Input arriving this soon after our own typing is our own echo.
const ECHO_GRACE_MS: u64 = 150;

static LAST_USER_MS: AtomicU64 = AtomicU64::new(0);
static TYPING_UNTIL_MS: AtomicU64 = AtomicU64::new(0);

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Called by the hotkey listener for key presses and clicks that are not the chord.
pub fn note_user_input() {
    note_user_input_at(now_ms());
}

pub fn note_user_input_at(now: u64) {
    if now > TYPING_UNTIL_MS.load(Ordering::SeqCst) {
        LAST_USER_MS.store(now, Ordering::SeqCst);
    }
}

pub fn begin_typing() {
    TYPING_UNTIL_MS.store(u64::MAX, Ordering::SeqCst);
}

pub fn end_typing() {
    TYPING_UNTIL_MS.store(now_ms() + ECHO_GRACE_MS, Ordering::SeqCst);
}

/// True when the user acted after `since_ms`.
pub fn user_active_since(since_ms: u64) -> bool {
    LAST_USER_MS.load(Ordering::SeqCst) > since_ms
}

/// What to do to the window so that `typed` becomes `target`: how many
/// characters to delete and what to type. Once `locked`, nothing is ever
/// deleted: the window is only extended when it is already a prefix of the target.
pub fn plan_sync(typed: &str, target: &str, locked: bool) -> (usize, String) {
    let common = typed
        .chars()
        .zip(target.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let extra = typed.chars().count() - common;
    if locked && extra > 0 {
        return (0, String::new());
    }
    (extra, target.chars().skip(common).collect())
}
