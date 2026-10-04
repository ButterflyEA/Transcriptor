use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::audio::{CpalRecorder, Recorder, Recording};
use crate::config::{Config, parse_model};
use crate::controller::{Action, Controller, Input, Status};
use crate::engine::{Engine, resolve_model_dir};
use crate::output::{PasteOutcome, Paster, system_paster};

pub enum RuntimeMsg {
    Input(Input),
    /// Internal: the max-recording timer for capture number `n` fired.
    AutoStop(u64),
    Shutdown,
}

pub struct Runtime {
    pub tx: Sender<RuntimeMsg>,
    handle: JoinHandle<()>,
}

impl Runtime {
    pub fn send(&self, input: Input) {
        let _ = self.tx.send(RuntimeMsg::Input(input));
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
    resolve_model_dir(config.model_dir.as_deref(), &exe_dir, &data_dir).ok_or_else(|| {
        format!(
            "model not found; looked in {}, {} and {}",
            config
                .model_dir
                .as_deref()
                .map_or("(no model_dir configured)".into(), |p| p
                    .display()
                    .to_string()),
            exe_dir.join("models").join("ivrit-turbo").display(),
            data_dir.join("models").join("ivrit-turbo").display(),
        )
    })
}

fn engine_thread(config: Config, jobs: mpsc::Receiver<Recording>, tx: Sender<RuntimeMsg>) {
    let send = |i: Input| {
        let _ = tx.send(RuntimeMsg::Input(i));
    };
    let loaded = parse_model(&config.model)
        .ok_or_else(|| format!("unknown model {:?}", config.model))
        .and_then(|size| {
            let dir = model_search_paths(&config)?;
            Engine::load(size, &dir).map_err(|e| e.to_string())
        });
    let engine = match loaded {
        Ok(e) => {
            send(Input::EngineReady);
            e
        }
        Err(e) => {
            send(Input::EngineFailed(e));
            return;
        }
    };
    let language = config.language_option();
    while let Ok(rec) = jobs.recv() {
        let result = engine
            .transcribe_pcm(&rec.samples, rec.sample_rate, language.as_deref())
            .map_err(|e| e.to_string());
        send(Input::Transcribed(result));
    }
}

/// Starts the coordinator (owns the controller and recorder) and the engine
/// thread (owns the model). Nothing here touches the network.
pub fn spawn(config: Config, on_status: StatusFn, on_notify: NotifyFn) -> Runtime {
    let recorder = CpalRecorder::new(config.input_device.clone(), config.max_record_secs);
    let paster = system_paster(config.paste_delay_ms);
    spawn_with(config, Box::new(recorder), paster, on_status, on_notify)
}

/// Like `spawn` but with injectable recorder and paster (used by tests).
pub fn spawn_with(
    config: Config,
    mut recorder: Box<dyn Recorder>,
    mut paster: Box<dyn Paster>,
    on_status: StatusFn,
    on_notify: NotifyFn,
) -> Runtime {
    let (tx, rx) = mpsc::channel::<RuntimeMsg>();
    let (job_tx, job_rx) = mpsc::channel::<Recording>();

    let engine_cfg = config.clone();
    let engine_tx = tx.clone();
    thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_thread(engine_cfg, job_rx, engine_tx))
        .expect("spawn engine thread");

    let timer_tx = tx.clone();
    let handle = thread::Builder::new()
        .name("coordinator".into())
        .spawn(move || {
            let mut controller = Controller::new(config.queue_cap);
            let mut capture_no = 0u64;
            let mut pending: VecDeque<Input> = VecDeque::new();
            on_status(Status::Loading);

            while let Ok(msg) = rx.recv() {
                match msg {
                    RuntimeMsg::Shutdown => break,
                    RuntimeMsg::AutoStop(n) if n == capture_no => pending.push_back(Input::Release),
                    RuntimeMsg::AutoStop(_) => {}
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
                                    let (n, tx, secs) =
                                        (capture_no, timer_tx.clone(), config.max_record_secs);
                                    thread::spawn(move || {
                                        thread::sleep(Duration::from_secs(secs as u64));
                                        let _ = tx.send(RuntimeMsg::AutoStop(n));
                                    });
                                }
                            }
                            Action::StopCapture => pending.push_back(match recorder.stop() {
                                Ok(r) => Input::Captured(r),
                                Err(e) => Input::CaptureFailed(e),
                            }),
                            Action::Transcribe(rec) => {
                                if job_tx.send(rec).is_err() {
                                    pending.push_back(Input::Transcribed(Err(
                                        "engine stopped".into()
                                    )));
                                }
                            }
                            Action::Paste(text) => match paster.paste(&text) {
                                Ok(PasteOutcome::Pasted) => {}
                                Ok(PasteOutcome::ClipboardOnly(why)) => {
                                    on_notify(format!("copied to clipboard, not pasted: {why}"))
                                }
                                Err(e) => on_notify(format!("paste failed: {e}")),
                            },
                            Action::SetStatus(s) => on_status(s),
                            Action::Notify(m) => on_notify(m),
                        }
                    }
                }
            }
        })
        .expect("spawn coordinator thread");

    Runtime { tx, handle }
}
