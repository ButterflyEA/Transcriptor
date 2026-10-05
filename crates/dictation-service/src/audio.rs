use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};

pub const MIN_RECORDING_MS: u64 = 300;

#[derive(Clone, Debug, PartialEq)]
pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Recording {
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        self.samples.len() as u64 * 1000 / self.sample_rate as u64
    }

    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0, |m, s| m.max(s.abs()))
    }
}

pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let n = channels.max(1) as usize;
    interleaved
        .chunks_exact(n)
        .map(|f| f.iter().sum::<f32>() / n as f32)
        .collect()
}

/// Appends `chunk` to `buf`, never letting `buf` grow beyond `max` samples.
pub fn push_capped(buf: &mut Vec<f32>, chunk: &[f32], max: usize) {
    let room = max.saturating_sub(buf.len());
    buf.extend_from_slice(&chunk[..chunk.len().min(room)]);
}

pub trait Recorder: Send {
    fn start(&mut self) -> Result<(), String>;
    /// Stops the recording started by `start` and returns what was captured.
    fn stop(&mut self) -> Result<Recording, String>;
}

enum Msg {
    Start(Sender<Result<(), String>>),
    Stop(Sender<Result<Recording, String>>),
}

/// Records from a cpal input device. The stream is `!Send`, so a dedicated
/// thread owns it and this handle talks to it over a channel.
pub struct CpalRecorder {
    tx: Sender<Msg>,
}

impl CpalRecorder {
    pub fn new(device_name: Option<String>, max_secs: u32) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || worker(rx, device_name, max_secs))
            .expect("spawn audio thread");
        Self { tx }
    }

    fn call<T>(&self, make: impl FnOnce(Sender<T>) -> Msg) -> Result<T, String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(make(reply_tx))
            .map_err(|_| "audio thread stopped".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "audio thread stopped".to_string())
    }
}

impl Recorder for CpalRecorder {
    fn start(&mut self) -> Result<(), String> {
        self.call(Msg::Start)?
    }

    fn stop(&mut self) -> Result<Recording, String> {
        self.call(Msg::Stop)?
    }
}

struct Active {
    _stream: cpal::Stream,
    buf: Arc<Mutex<Vec<f32>>>,
    rate: u32,
    /// Set by the stream error callback, e.g. when the microphone is unplugged.
    lost: Arc<AtomicBool>,
}

fn worker(rx: Receiver<Msg>, device_name: Option<String>, max_secs: u32) {
    let mut active: Option<Active> = None;
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Start(reply) => {
                active = None;
                let result = open(device_name.as_deref(), max_secs).map(|a| {
                    active = Some(a);
                });
                let _ = reply.send(result);
            }
            Msg::Stop(reply) => {
                let result = match active.take() {
                    None => Err("not recording".to_string()),
                    Some(a) if a.lost.load(Ordering::SeqCst) => {
                        Err("microphone disconnected during recording".to_string())
                    }
                    Some(a) => {
                        drop(a._stream);
                        let samples = std::mem::take(&mut *a.buf.lock().unwrap());
                        Ok(Recording {
                            samples,
                            sample_rate: a.rate,
                        })
                    }
                };
                let _ = reply.send(result);
            }
        }
    }
}

fn pick_device(name: Option<&str>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    if let Some(name) = name {
        if let Ok(mut devices) = host.input_devices()
            && let Some(d) = devices.find(|d| d.name().ok().as_deref() == Some(name))
        {
            return Ok(d);
        }
        log::warn!("input device {name:?} not found, using default");
    }
    host.default_input_device()
        .ok_or_else(|| "no input device".to_string())
}

fn open(name: Option<&str>, max_secs: u32) -> Result<Active, String> {
    let device = pick_device(name)?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("input config: {e}"))?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels();
    let max = rate as usize * max_secs as usize;
    let buf = Arc::new(Mutex::new(Vec::new()));
    let lost = Arc::new(AtomicBool::new(false));
    let config = supported.config();

    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config, channels, max, &buf, &lost),
        SampleFormat::I16 => build::<i16>(&device, &config, channels, max, &buf, &lost),
        SampleFormat::U16 => build::<u16>(&device, &config, channels, max, &buf, &lost),
        other => Err(format!("unsupported sample format {other:?}")),
    }?;
    stream.play().map_err(|e| format!("start stream: {e}"))?;
    Ok(Active {
        _stream: stream,
        buf,
        rate,
        lost,
    })
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    max: usize,
    buf: &Arc<Mutex<Vec<f32>>>,
    lost: &Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let buf = Arc::clone(buf);
    let lost = Arc::clone(lost);
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let floats: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
                let mono = downmix_to_mono(&floats, channels);
                push_capped(&mut buf.lock().unwrap(), &mono, max);
            },
            move |e| {
                log::error!("audio stream error: {e}");
                lost.store(true, Ordering::SeqCst);
            },
            None,
        )
        .map_err(|e| format!("open input stream: {e}"))
}
