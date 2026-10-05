use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio::{CpalRecorder, LiveTap, Recorder, Recording};
use crate::config::{Config, parse_model};
use crate::controller::{Action, Controller, Input, Status};
use crate::engine::{Engine, resolve_model_dir};
use crate::output::{Paster, system_paster};
use crate::outworker::{Job as OutJob, OutWorker};
use crate::stream::{EngineDecoder, StreamEvent, StreamLoop, StreamParams};
use crate::typer::{NullTyper, Typer};

pub enum RuntimeMsg {
    Input(Input),
    /// Internal: the max-recording timer for capture number `n` fired.
    AutoStop(u64),
    /// Press if idle, release if recording.
    Toggle,
    /// Apply a freshly loaded config: reloads the model only if it changed.
    Reload(Box<Config>),
    /// While paused, new recordings are refused (a recording in progress can still stop).
    SetPaused(bool),
    /// Internal: a live-transcription event from the engine thread.
    Stream(StreamEvent),
    Shutdown,
}

/// True when a message would start a recording and must be ignored while paused.
pub fn blocked_by_pause(paused: bool, msg: &RuntimeMsg) -> bool {
    paused && matches!(msg, RuntimeMsg::Input(Input::Press) | RuntimeMsg::Toggle)
}

enum Job {
    Transcribe(Recording),
    /// Live-transcribe a capture in progress until stop is set.
    Stream {
        tap: LiveTap,
        capture_id: u64,
        stop: Arc<AtomicBool>,
    },
    Reload(Box<Config>),
}

pub struct Runtime {
    pub tx: Sender<RuntimeMsg>,
    status: Arc<Mutex<Status>>,
    handle: JoinHandle<()>,
}

impl Runtime {
    pub fn send(&self, input: Input) {
        let _ = self.tx.send(RuntimeMsg::Input(input));
    }

    /// Shared handle to the latest status, for the control server.
    pub fn status_handle(&self) -> Arc<Mutex<Status>> {
        Arc::clone(&self.status)
    }

    pub fn status(&self) -> Status {
        *self.status.lock().unwrap()
    }

    pub fn shutdown(self) {
        let _ = self.tx.send(RuntimeMsg::Shutdown);
        let _ = self.handle.join();
    }
}

pub type StatusFn = Box<dyn Fn(Status) + Send>;
pub type NotifyFn = Box<dyn Fn(String) + Send>;

fn model_search_paths(config: &Config) -> Result<PathBuf, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_default();
    let data_dir = dirs::data_dir().unwrap_or_default().join("dictation");
    resolve_model_dir(
        &config.model,
        config.model_dir.as_deref(),
        &exe_dir,
        &data_dir,
    )
    .ok_or_else(|| {
        format!(
            "model not found; looked in {}, {} and {}",
            config
                .model_dir
                .as_deref()
                .map_or("(no model_dir configured)".into(), |p| p
                    .display()
                    .to_string()),
            exe_dir.join("models").join(&config.model).display(),
            data_dir.join("models").join(&config.model).display(),
        ) + if cfg!(target_os = "linux") {
            format!(" and /usr/share/dictation/models/{}", config.model)
        } else {
            String::new()
        }
        .as_str()
    })
}

fn load_engine(config: &Config) -> Result<Engine, String> {
    let size =
        parse_model(&config.model).ok_or_else(|| format!("unknown model {:?}", config.model))?;
    let dir = model_search_paths(config)?;
    Engine::load(size, &dir, config.precision).map_err(|e| e.to_string())
}

fn engine_thread(config: Config, jobs: mpsc::Receiver<Job>, tx: Sender<RuntimeMsg>) {
    let send = |i: Input| {
        let _ = tx.send(RuntimeMsg::Input(i));
    };
    let mut config = config;
    let mut engine = match load_engine(&config) {
        Ok(e) => {
            send(Input::EngineReady);
            Some(e)
        }
        Err(e) => {
            send(Input::EngineFailed(e));
            None
        }
    };
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Transcribe(rec) => {
                let result = match &engine {
                    Some(e) => e
                        .transcribe_pcm(
                            &rec.samples,
                            rec.sample_rate,
                            config.language_option().as_deref(),
                        )
                        .map_err(|e| e.to_string()),
                    None => Err("model is not loaded".to_string()),
                };
                send(Input::Transcribed(result));
            }
            Job::Stream {
                tap,
                capture_id,
                stop,
            } => {
                let Some(e) = &engine else { continue };
                let params = StreamParams::from_config(&config);
                let decoder = EngineDecoder {
                    engine: e,
                    language: config.language_option(),
                    window_samples: params.window_samples,
                };
                StreamLoop::new(decoder, params).run(tap, capture_id, stop, |ev| {
                    let _ = tx.send(RuntimeMsg::Stream(ev));
                });
            }
            Job::Reload(new) => {
                let model_changed = engine.is_none()
                    || new.model != config.model
                    || new.model_dir != config.model_dir;
                config = *new;
                if model_changed {
                    // Free the old weights before loading, so VRAM is never held twice.
                    engine = None;
                    engine = match load_engine(&config) {
                        Ok(e) => Some(e),
                        Err(e) => {
                            send(Input::EngineFailed(e));
                            continue;
                        }
                    };
                }
                send(Input::EngineReady);
            }
        }
    }
}
/// Starts the coordinator (owns the controller and recorder) and the engine
/// thread (owns the model). Nothing here touches the network.
pub fn spawn(config: Config, on_status: StatusFn, on_notify: NotifyFn) -> Runtime {
    let recorder = CpalRecorder::new(config.input_device.clone(), config.max_record_secs);
    let paster = system_paster(config.paste_delay_ms);
    spawn_with(config, Box::new(recorder), paster, on_status, on_notify)
}

/// Like `spawn_with_typer`, with typing disabled.
pub fn spawn_with(
    config: Config,
    recorder: Box<dyn Recorder>,
    paster: Box<dyn Paster>,
    on_status: StatusFn,
    on_notify: NotifyFn,
) -> Runtime {
    spawn_with_typer(
        config,
        recorder,
        paster,
        Box::new(NullTyper),
        on_status,
        on_notify,
    )
}

/// Like `spawn` but with injectable recorder, paster and typer (used by tests).
pub fn spawn_with_typer(
    config: Config,
    mut recorder: Box<dyn Recorder>,
    paster: Box<dyn Paster>,
    typer: Box<dyn Typer>,
    on_status: StatusFn,
    on_notify: NotifyFn,
) -> Runtime {
    let (tx, rx) = mpsc::channel::<RuntimeMsg>();
    let (job_tx, job_rx) = mpsc::channel::<Job>();

    let engine_cfg = config.clone();
    let engine_tx = tx.clone();
    thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_thread(engine_cfg, job_rx, engine_tx))
        .expect("spawn engine thread");

    let status = Arc::new(Mutex::new(Status::Loading));
    let status_w = Arc::clone(&status);
    let timer_tx = tx.clone();
    let on_notify = Arc::new(Mutex::new(on_notify));
    let worker_notify = Arc::clone(&on_notify);
    let out = OutWorker::spawn(
        paster,
        typer,
        Box::new(move |m| (worker_notify.lock().unwrap())(m)),
    );
    let on_notify = move |m: String| (on_notify.lock().unwrap())(m);
    let handle = thread::Builder::new()
        .name("coordinator".into())
        .spawn(move || {
            let mut controller = Controller::new(config.queue_cap);
            let mut capture_no = 0u64;
            let mut pending: VecDeque<Input> = VecDeque::new();
            let mut paused = false;
            let mut stream_stop: Option<Arc<AtomicBool>> = None;
            on_status(Status::Loading);

            while let Ok(msg) = rx.recv() {
                if blocked_by_pause(paused, &msg) && !controller.is_recording() {
                    continue;
                }
                match msg {
                    RuntimeMsg::SetPaused(p) => paused = p,
                    RuntimeMsg::Shutdown => break,
                    RuntimeMsg::AutoStop(n) if n == capture_no => pending.push_back(Input::Release),
                    RuntimeMsg::AutoStop(_) => {}
                    RuntimeMsg::Toggle => pending.push_back(if controller.is_recording() {
                        Input::Release
                    } else {
                        Input::Press
                    }),
                    RuntimeMsg::Reload(cfg) => {
                        for action in controller.set_loading() {
                            if let Action::SetStatus(s) = action {
                                *status_w.lock().unwrap() = s;
                                on_status(s);
                            }
                        }
                        let _ = job_tx.send(Job::Reload(cfg));
                    }
                    // Stream mode is tap-to-toggle: a press ends a recording in progress and the release is ignored.
                    RuntimeMsg::Input(Input::Press) if config.stream && controller.is_recording() => {
                        pending.push_back(Input::Release)
                    }
                    RuntimeMsg::Input(Input::Release) if config.stream => {}
                    RuntimeMsg::Stream(StreamEvent::Edit { capture_id, edit }) => {
                        eprintln!("stream[{capture_id}] {edit:?}");
                    }
                    RuntimeMsg::Stream(StreamEvent::Slow { pass_ms }) => {
                        on_notify(format!(
                            "Streaming is too slow on this machine ({pass_ms} ms per pass); use normal dictation."
                        ));
                    }
                    RuntimeMsg::Input(i) => pending.push_back(i),
                }
                while let Some(input) = pending.pop_front() {
                    for action in controller.handle(input) {
                        match action {
                            Action::StartCapture => {
                                capture_no += 1;
                                if let Err(e) = recorder.start() {
                                    pending.push_back(Input::CaptureFailed(e));
                                } else {
                                    if config.stream && let Some(tap) = recorder.tap() {
                                        let stop = Arc::new(AtomicBool::new(false));
                                        stream_stop = Some(stop.clone());
                                        let _ = job_tx.send(Job::Stream {
                                            tap,
                                            capture_id: capture_no,
                                            stop,
                                        });
                                    }
                                    let (n, tx, secs) =
                                        (capture_no, timer_tx.clone(), config.max_record_secs);
                                    thread::spawn(move || {
                                        thread::sleep(Duration::from_secs(secs as u64));
                                        let _ = tx.send(RuntimeMsg::AutoStop(n));
                                    });
                                }
                            }
                            Action::StopCapture => {
                                if let Some(stop) = stream_stop.take() {
                                    stop.store(true, Ordering::SeqCst);
                                }
                                pending.push_back(match recorder.stop() {
                                Ok(r) => Input::Captured(r),
                                Err(e) => Input::CaptureFailed(e),
                                })
                            }
                            Action::Transcribe(rec) => {
                                if job_tx.send(Job::Transcribe(rec)).is_err() {
                                    pending.push_back(Input::Transcribed(Err(
                                        "engine stopped".into()
                                    )));
                                }
                            }
                            Action::Paste(text) => out.send(OutJob::Paste(text)),
                            Action::SetStatus(s) => {
                                *status_w.lock().unwrap() = s;
                                on_status(s);
                            }
                            Action::Notify(m) => on_notify(m),
                        }
                    }
                }
            }
            out.shutdown();
        })
        .expect("spawn coordinator thread");

    Runtime { tx, status, handle }
}
