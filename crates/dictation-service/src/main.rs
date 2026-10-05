use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use dictation_service::audio::{CpalRecorder, Recorder};
use dictation_service::checksum::FileCheck;
use dictation_service::chord::{Chord, HotkeyEvent};
use dictation_service::config::{self, Config, Precision, parse_model};
use dictation_service::control::{self, Command};
use dictation_service::controller::Input;
use dictation_service::engine::Engine;
use dictation_service::hotkey;
use dictation_service::output::{PasteOutcome, system_paster};
use dictation_service::runtime::{RuntimeMsg, spawn};
use dictation_service::tray::{self, UiEvent};
use whisper_burn::audio::decode::decode_to_mono_f32;
use whisper_burn::backends::{BackendChoice, cpu_device, default_device};

#[derive(Parser)]
#[command(
    name = "dictation",
    version,
    about = "Push-to-talk dictation with whisper-burn"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Auto,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the dictation service in the foreground (default).
    Run,
    /// Start recording (talk to the running service).
    Start,
    /// Stop recording and transcribe.
    Stop,
    /// Start recording if idle, otherwise stop.
    Toggle,
    /// Print the service status: loading, idle, recording, transcribing or error.
    Status,
    /// Re-read the config file (reloads the model only if it changed).
    Reload,
    /// Open the settings window.
    Settings,
    /// Check the model files against SHA256SUMS (for transfers into airgapped networks).
    #[command(hide = true)]
    VerifyModel {
        /// Model folder; defaults to the folder the service would use.
        dir: Option<PathBuf>,
    },
    /// Developer tool: copy text to the clipboard and paste it after 3 seconds.
    #[command(hide = true)]
    PasteTest { text: String },
    /// Developer tool: record from the microphone and print level stats.
    #[command(hide = true)]
    RecordTest {
        #[arg(long, default_value_t = 3)]
        secs: u64,
        #[arg(long)]
        device: Option<String>,
        /// Also save the capture as a 16-bit mono WAV file.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Developer tool: transcribe an audio file from a local model folder.
    #[command(hide = true)]
    TranscribeFile {
        path: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "ivrit-turbo")]
        model: String,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, value_enum, default_value_t = Device::Auto)]
        device: Device,
        /// Transcribe this many times to separate warm-up from steady state.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        #[arg(long, value_enum, default_value = "f16")]
        precision: Precision,
    },
    /// Developer tool: measure short-window decoding against the 30 s decode.
    #[command(hide = true)]
    StreamSweep {
        path: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "he")]
        language: String,
        /// Window lengths in seconds, comma separated.
        #[arg(long, value_delimiter = ',', default_value = "1,2,3,5,10")]
        windows: Vec<f32>,
        #[arg(long, value_enum, default_value = "f16")]
        precision: Precision,
        /// Zero-pad each window to 30 s before encoding.
        #[arg(long)]
        pad: bool,
    },
}

fn main() -> ExitCode {
    match Args::parse().command {
        None | Some(Cmd::Run) => report(run()),
        Some(Cmd::Start) => control_cmd(Command::Start),
        Some(Cmd::Stop) => control_cmd(Command::Stop),
        Some(Cmd::Toggle) => control_cmd(Command::Toggle),
        Some(Cmd::Status) => control_cmd(Command::Status),
        Some(Cmd::Reload) => control_cmd(Command::Reload),
        Some(Cmd::VerifyModel { dir }) => verify_model(dir),
        Some(Cmd::Settings) => report(dictation_service::settings::run()),
        Some(Cmd::PasteTest { text }) => report(paste_test(&text)),
        Some(Cmd::RecordTest { secs, device, out }) => match record_test(secs, device, out) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some(Cmd::StreamSweep {
            path,
            model_dir,
            language,
            windows,
            precision,
            pad,
        }) => match stream_sweep(&path, &model_dir, &language, &windows, precision, pad) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some(Cmd::TranscribeFile {
            path,
            model_dir,
            model,
            language,
            device,
            repeat,
            precision,
        }) => match transcribe_file(
            &path, &model_dir, &model, language, device, repeat, precision,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
    }
}

fn verify_model(dir: Option<PathBuf>) -> ExitCode {
    let dir = dir.unwrap_or_else(|| {
        let cfg = Config::load_from(&config::default_path()).unwrap_or_default();
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_default();
        let data_dir = dirs::data_dir().unwrap_or_default().join("dictation");
        dictation_service::engine::resolve_model_dir(
            &cfg.model,
            cfg.model_dir.as_deref(),
            &exe_dir,
            &data_dir,
        )
        .unwrap_or_else(|| exe_dir.join("models").join(&cfg.model))
    });
    match dictation_service::checksum::verify_dir(&dir) {
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
        Ok(results) => {
            let mut all_ok = true;
            for (name, check) in &results {
                match check {
                    FileCheck::Ok => println!("OK        {name}"),
                    FileCheck::Missing => {
                        all_ok = false;
                        println!("MISSING   {name}");
                    }
                    FileCheck::Mismatch { actual } => {
                        all_ok = false;
                        println!("MISMATCH  {name} (got {actual})");
                    }
                }
            }
            if all_ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

fn transcribe_file(
    path: &Path,
    model_dir: &Path,
    model: &str,
    language: Option<String>,
    device: Device,
    repeat: u32,
    precision: Precision,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = parse_model(model).ok_or("unknown model")?;
    let (pcm, rate) = decode_to_mono_f32(path)?;
    let mut precision = precision;
    let choice = match device {
        Device::Auto => {
            precision = Engine::resolve_precision(precision);
            default_device()
        }
        Device::Cpu => BackendChoice::Cpu(cpu_device()),
    };

    let t = Instant::now();
    let engine = Engine::load_on(size, model_dir, choice, precision)?;
    println!(
        "backend={} load_ms={}",
        engine.backend_name(),
        t.elapsed().as_millis()
    );
    println!("audio_s={:.1}", pcm.len() as f64 / rate as f64);

    for run in 1..=repeat {
        let t = Instant::now();
        let text = engine.transcribe_pcm(&pcm, rate, language.as_deref())?;
        println!("run={run} elapsed_ms={}", t.elapsed().as_millis());
        println!("{text}");
    }
    Ok(())
}

fn record_test(secs: u64, device: Option<String>, out: Option<PathBuf>) -> Result<(), String> {
    let mut rec = CpalRecorder::new(device, 120);
    rec.start()?;
    std::thread::sleep(std::time::Duration::from_secs(secs));
    let r = rec.stop()?;
    println!("samples={}", r.samples.len());
    println!("rate={}", r.sample_rate);
    println!("duration_ms={}", r.duration_ms());
    println!("peak={:.4}", r.peak());
    if let Some(path) = out {
        write_wav(&path, &r.samples, r.sample_rate).map_err(|e| e.to_string())?;
        println!("saved={}", path.display());
    }
    Ok(())
}

fn write_wav(path: &Path, samples: &[f32], rate: u32) -> std::io::Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b)
}

fn report(result: Result<(), Box<dyn std::error::Error>>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn paste_test(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("focus a text box; pasting in 3 seconds...");
    std::thread::sleep(std::time::Duration::from_secs(3));
    match system_paster(50).paste(text)? {
        PasteOutcome::Pasted => eprintln!("pasted"),
        PasteOutcome::ClipboardOnly(why) => eprintln!("clipboard only: {why}"),
    }
    Ok(())
}

fn control_cmd(cmd: Command) -> ExitCode {
    match control::send(cmd) {
        Ok(reply) => {
            println!("{reply}");
            if reply.starts_with("error") {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("dictation is not running ({e}). Start it with `dictation run`.");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    if control::send(Command::Status).is_ok() {
        return Err("dictation is already running".into());
    }
    let config: Config = Config::load_from(&config::default_path())?;
    config.validate()?;
    let hotkey_label = config.hotkey.clone();
    let chord = Chord::parse(&hotkey_label)?;
    let event_loop = tray::event_loop();
    let (status_proxy, notice_proxy) = (event_loop.create_proxy(), event_loop.create_proxy());
    let rt = spawn(
        config,
        Box::new(move |s| {
            eprintln!("status: {s:?}");
            let _ = status_proxy.send_event(UiEvent::Status(s));
        }),
        Box::new(move |m| {
            eprintln!("notice: {m}");
            let _ = notice_proxy.send_event(UiEvent::Notice(m));
        }),
    );
    start_hotkey(chord, &hotkey_label, rt.tx.clone());
    let tx = rt.tx.clone();
    let status = rt.status_handle();
    // The control server runs beside the tray loop; Quit ends the process.
    std::thread::Builder::new()
        .name("control".into())
        .spawn(move || {
            let result = control::serve(move |cmd| match cmd {
                Command::Start => send_input(&tx, Input::Press),
                Command::Stop => send_input(&tx, Input::Release),
                Command::Toggle => match tx.send(RuntimeMsg::Toggle) {
                    Ok(()) => "ok".into(),
                    Err(_) => "error: service is shutting down".into(),
                },
                Command::Status => status.lock().unwrap().as_str().to_string(),
                Command::Reload => match Config::load_from(&config::default_path())
                    .and_then(|c| c.validate().map(|()| c))
                {
                    Ok(c) => match tx.send(RuntimeMsg::Reload(Box::new(c))) {
                        Ok(()) => "ok".into(),
                        Err(_) => "error: service is shutting down".into(),
                    },
                    Err(e) => format!("error: {e}"),
                },
            });
            if let Err(e) = result {
                eprintln!("control server stopped: {e}");
            }
        })?;
    tray::run(event_loop, rt)
}
fn send_input(tx: &std::sync::mpsc::Sender<RuntimeMsg>, input: Input) -> String {
    match tx.send(RuntimeMsg::Input(input)) {
        Ok(()) => "ok".into(),
        Err(_) => "error: service is shutting down".into(),
    }
}

/// Global hotkey: hold to record, release to transcribe. Failure is not fatal,
/// since `dictation start/stop` still works.
fn start_hotkey(chord: Chord, label: &str, tx: std::sync::mpsc::Sender<RuntimeMsg>) {
    let Some(backend) = hotkey::select() else {
        eprintln!(
            "no global hotkey backend on this session; bind `dictation start`/`stop` in your compositor (see docs/dictation.md)"
        );
        return;
    };
    let name = backend.name();
    let result = backend.spawn(
        chord,
        Box::new(move |e| {
            let input = match e {
                HotkeyEvent::Pressed => Input::Press,
                HotkeyEvent::Released => Input::Release,
            };
            let _ = tx.send(RuntimeMsg::Input(input));
        }),
    );
    match result {
        Ok(()) => eprintln!("hotkey {label} active via {name}"),
        Err(e) => eprintln!("hotkey unavailable ({e}); use `dictation start`/`stop`"),
    }
}

fn stream_sweep(
    path: &Path,
    model_dir: &Path,
    language: &str,
    windows: &[f32],
    precision: Precision,
    pad: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use dictation_service::engine::join_segments;
    use dictation_service::metrics::char_error_rate;

    let (pcm, rate) = decode_to_mono_f32(path)?;
    let pcm16 = if rate == 16_000 {
        pcm
    } else {
        whisper_burn::audio::resample::resample_to_16k(&pcm, rate, 16_000)?
    };
    let precision = Engine::resolve_precision(precision);
    let engine = Engine::load_on(
        parse_model("ivrit-turbo").ok_or("model")?,
        model_dir,
        default_device(),
        precision,
    )?;
    println!("backend={} audio_s={:.1}", engine.backend_name(), pcm16.len() as f64 / 16_000.0);

    // Warm up so the first window length is not penalised by shader compilation.
    engine.transcribe_windows(&pcm16[..16_000.min(pcm16.len())], 16_000, Some(language), Some(16_000), pad)?;

    let t = Instant::now();
    let reference = join_segments(&engine.transcribe_windows(&pcm16, 16_000, Some(language), None, false)?);
    println!("full30 elapsed_ms={}", t.elapsed().as_millis());
    println!("REF: {reference}");

    println!("window_s | windows | ms/window | CER vs full");
    for &secs in windows {
        let w = ((secs * 16_000.0) as usize / 320).max(1) * 320;
        let t = Instant::now();
        let segs = engine.transcribe_windows(&pcm16, 16_000, Some(language), Some(w), pad)?;
        let ms = t.elapsed().as_millis();
        let n = pcm16.len().div_ceil(w);
        let text = join_segments(&segs);
        println!(
            "{secs:>8} | {n:>7} | {:>9} | {:.3}",
            ms / n as u128,
            char_error_rate(&reference, &text)
        );
        println!("  {text}");
    }
    Ok(())
}