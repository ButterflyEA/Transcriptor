use std::collections::VecDeque;

use crate::audio::{MIN_RECORDING_MS, Recording};

#[derive(Debug)]
pub enum Input {
    Press,
    Release,
    Captured(Recording),
    CaptureFailed(String),
    Transcribed(Result<String, String>),
    EngineReady,
    EngineFailed(String),
}

#[derive(Debug)]
pub enum Action {
    StartCapture,
    StopCapture,
    Transcribe(Recording),
    Paste(String),
    SetStatus(Status),
    Notify(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Loading,
    Idle,
    Recording,
    Transcribing,
    Error,
}

pub struct Controller {
    ready: bool,
    recording: bool,
    busy: bool,
    queue: VecDeque<Recording>,
    queue_cap: usize,
}

impl Controller {
    pub fn new(queue_cap: usize) -> Self {
        Self {
            ready: false,
            recording: false,
            busy: false,
            queue: VecDeque::new(),
            queue_cap,
        }
    }

    fn status(&self) -> Status {
        if !self.ready {
            Status::Loading
        } else if self.recording {
            Status::Recording
        } else if self.busy {
            Status::Transcribing
        } else {
            Status::Idle
        }
    }

    fn start_next(&mut self, out: &mut Vec<Action>) {
        if self.busy {
            return;
        }
        if let Some(job) = self.queue.pop_front() {
            self.busy = true;
            out.push(Action::Transcribe(job));
        }
    }

    pub fn handle(&mut self, input: Input) -> Vec<Action> {
        let mut out = Vec::new();
        match input {
            Input::EngineReady => self.ready = true,
            Input::EngineFailed(m) => {
                out.push(Action::SetStatus(Status::Error));
                out.push(Action::Notify(format!("model failed to load: {m}")));
                return out;
            }
            Input::Press if !self.ready => {
                return vec![Action::Notify("model still loading".into())];
            }
            Input::Press if self.recording => return Vec::new(),
            Input::Press => {
                self.recording = true;
                out.push(Action::StartCapture);
            }
            Input::Release if !self.recording => return Vec::new(),
            Input::Release => out.push(Action::StopCapture),
            Input::CaptureFailed(m) => {
                self.recording = false;
                out.push(Action::Notify(format!("recording failed: {m}")));
            }
            Input::Captured(rec) => {
                self.recording = false;
                if rec.duration_ms() >= MIN_RECORDING_MS {
                    if self.queue.len() >= self.queue_cap {
                        out.push(Action::Notify(
                            "transcription queue full; recording dropped".into(),
                        ));
                    } else {
                        self.queue.push_back(rec);
                        self.start_next(&mut out);
                    }
                }
            }
            Input::Transcribed(result) => {
                self.busy = false;
                match result {
                    Ok(text) if !text.trim().is_empty() => out.push(Action::Paste(text)),
                    Ok(_) => {}
                    Err(m) => out.push(Action::Notify(format!("transcription failed: {m}"))),
                }
                self.start_next(&mut out);
            }
        }
        out.push(Action::SetStatus(self.status()));
        out
    }
}
