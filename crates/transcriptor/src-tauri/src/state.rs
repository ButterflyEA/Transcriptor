use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use whisper_burn::TranscriptionSegment;

/// Process-shared state: guards one run at a time and keeps the last result
/// so save/copy commands can render without a live pipeline.
#[derive(Default)]
pub struct AppState {
    pub running: AtomicBool,
    pub cancel: Arc<AtomicBool>,
    pub last_segments: Mutex<Option<Vec<TranscriptionSegment>>>,
    /// Language the kept segments were actually decoded in. This is the
    /// *resolved* one, not the requested one: the app offers "auto" detection,
    /// so echoing the request would mislabel every detected export.
    pub last_language: Mutex<Option<String>>,
}

impl AppState {
    pub fn snapshot(&self) -> Vec<TranscriptionSegment> {
        self.last_segments.lock().unwrap().clone().unwrap_or_default()
    }

    pub fn language(&self) -> Option<String> {
        self.last_language.lock().unwrap().clone()
    }
}