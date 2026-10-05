//! Engine-side loop that decodes the growing recording in short windows while
//! the user is still speaking, and turns the passes into typing edits.
//!
//! Audio is cut into windows of `window_samples`; consecutive windows overlap
//! by `overlap_samples`. A completed window is final and is stitched onto the
//! text so far, the window being recorded is re-decoded every pass.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::audio::LiveTap;
use crate::config::Config;
use crate::engine::Engine;
use crate::stitch::stitch;
use crate::transcript::{Edit, Transcript, words};
use crate::vad::{DEFAULT_RMS, is_speech};

const RATE: u32 = 16_000;
/// Shortest stretch of audio worth decoding.
const MIN_SAMPLES: usize = 8_000;
const MAX_STITCH_WORDS: usize = 4;
/// A pass slower than this many windows means the GPU cannot keep up.
const SLOW_FACTOR: u64 = 3;

pub trait WindowDecoder {
    /// Decodes up to one window of 16 kHz audio.
    fn decode(&self, pcm: &[f32], cancelled: &Arc<AtomicBool>) -> Result<String, String>;
}

pub struct EngineDecoder<'a> {
    pub engine: &'a Engine,
    pub language: Option<String>,
    pub window_samples: usize,
}

impl WindowDecoder for EngineDecoder<'_> {
    fn decode(&self, pcm: &[f32], cancelled: &Arc<AtomicBool>) -> Result<String, String> {
        self.engine
            .decode_window(pcm, self.language.as_deref(), self.window_samples, cancelled)
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Edit { capture_id: u64, edit: Edit },
    /// A pass took more than three windows; the machine is too slow to keep up.
    Slow { pass_ms: u64 },
}

#[derive(Clone, Debug)]
pub struct StreamParams {
    pub window_samples: usize,
    pub overlap_samples: usize,
    pub interval: Duration,
    pub lock_words: usize,
    pub rms_threshold: f32,
}

impl StreamParams {
    pub fn from_config(c: &Config) -> Self {
        Self {
            window_samples: c.stream_window_secs as usize * RATE as usize,
            overlap_samples: c.stream_overlap_ms as usize * RATE as usize / 1000,
            interval: Duration::from_millis(c.stream_interval_ms as u64),
            lock_words: c.stream_lock_words,
            rms_threshold: DEFAULT_RMS,
        }
    }
}

pub struct StreamLoop<D> {
    decoder: D,
    params: StreamParams,
    transcript: Transcript,
    /// Words of completed windows, stitched together.
    finalised: Vec<String>,
    /// Start of the window being recorded, in 16 kHz samples.
    win_start: usize,
    interval: Duration,
    slow_reported: bool,
}

impl<D: WindowDecoder> StreamLoop<D> {
    pub fn new(decoder: D, params: StreamParams) -> Self {
        Self {
            transcript: Transcript::new(params.lock_words),
            interval: params.interval,
            decoder,
            params,
            finalised: Vec::new(),
            win_start: 0,
            slow_reported: false,
        }
    }

    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }

    /// Runs passes until `stop` is set. Passes never overlap in time: the next
    /// one starts when the previous has finished and the interval has passed.
    pub fn run(
        &mut self,
        tap: LiveTap,
        capture_id: u64,
        stop: Arc<AtomicBool>,
        mut emit: impl FnMut(StreamEvent),
    ) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        // Aborts the pass in flight as soon as stop is requested.
        let watcher = {
            let (stop, cancelled, done) = (stop.clone(), cancelled.clone(), done.clone());
            thread::spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    if stop.load(Ordering::SeqCst) {
                        cancelled.store(true, Ordering::SeqCst);
                        return;
                    }
                    thread::sleep(Duration::from_millis(5));
                }
            })
        };

        let mut last_len = 0usize;
        let mut last_start = Instant::now() - self.interval;
        while !stop.load(Ordering::SeqCst) {
            if last_start.elapsed() < self.interval {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            last_start = Instant::now();
            let Some(pcm) = self.window_audio(&tap) else {
                continue;
            };
            if pcm.len() < MIN_SAMPLES || pcm.len() == last_len {
                continue;
            }
            let complete = pcm.len() >= self.params.window_samples;
            let segment = &pcm[..pcm.len().min(self.params.window_samples)];
            let text = if is_speech(segment, RATE, self.params.rms_threshold) {
                match self.decoder.decode(segment, &cancelled) {
                    Ok(t) => t,
                    Err(e) => {
                        if !stop.load(Ordering::SeqCst) {
                            log::warn!("stream pass failed: {e}");
                        }
                        continue;
                    }
                }
            } else {
                String::new()
            };
            if stop.load(Ordering::SeqCst) {
                break;
            }
            let pass_ms = last_start.elapsed().as_millis() as u64;
            self.pace(pass_ms, &mut emit);

            let pass_words = words(&text);
            let full = if complete {
                self.complete_window(pass_words);
                last_len = 0;
                self.finalised.clone()
            } else {
                last_len = pcm.len();
                stitch(&self.finalised, &pass_words, MAX_STITCH_WORDS)
            };
            let locked = self.transcript.locked_words();
            let tail = full.get(locked..).unwrap_or(&[]).join(" ");
            if let Some(edit) = self.transcript.on_pass(&tail) {
                emit(StreamEvent::Edit { capture_id, edit });
            }
        }
        done.store(true, Ordering::SeqCst);
        let _ = watcher.join();
    }

    /// Audio of the window being recorded, resampled to 16 kHz.
    fn window_audio(&self, tap: &LiveTap) -> Option<Vec<f32>> {
        let rate = tap.rate();
        let offset = (self.win_start as u64 * rate as u64 / RATE as u64) as usize;
        let (raw, rate, _) = tap.snapshot_from(offset);
        if rate == RATE {
            return Some(raw);
        }
        match whisper_burn::audio::resample::resample_to_16k(&raw, rate, RATE) {
            Ok(p) => Some(p),
            Err(e) => {
                log::warn!("stream resample failed: {e}");
                None
            }
        }
    }

    /// Adds a finished window. Its last words lie in the overlap and may be cut
    /// mid-word, so they are left to the next window, which hears them whole.
    fn complete_window(&mut self, mut w: Vec<String>) {
        let step = self.params.window_samples - self.params.overlap_samples;
        let held = (w.len() * self.params.overlap_samples).div_ceil(self.params.window_samples);
        w.truncate(w.len() - held.min(w.len()));
        self.finalised = stitch(&self.finalised, &w, MAX_STITCH_WORDS);
        self.win_start += step;
    }

    fn pace(&mut self, pass_ms: u64, emit: &mut impl FnMut(StreamEvent)) {
        let window_ms = self.params.window_samples as u64 * 1000 / RATE as u64;
        if pass_ms > window_ms * SLOW_FACTOR && !self.slow_reported {
            self.slow_reported = true;
            emit(StreamEvent::Slow { pass_ms });
        }
        self.interval = self.interval.max(Duration::from_millis(pass_ms));
    }
}
