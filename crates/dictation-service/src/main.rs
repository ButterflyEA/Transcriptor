use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use dictation_service::audio::{CpalRecorder, Recorder};
use dictation_service::config::{self, Config, parse_model};
use dictation_service::controller::Input;
use dictation_service::engine::Engine;
use dictation_service::output::{PasteOutcome, system_paster};
use dictation_service::runtime::spawn;
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
    /// Run the dictation service in the foreground (default). Until global
    /// hotkeys arrive, type `s` + Enter to start, `e` + Enter to stop, `q` to quit.
    Run,
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
    },
}

fn main() -> ExitCode {
    match Args::parse().command {
        None | Some(Cmd::Run) => report(run()),
        Some(Cmd::PasteTest { text }) => report(paste_test(&text)),
        Some(Cmd::RecordTest { secs, device }) => match record_test(secs, device) {
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
        }) => match transcribe_file(&path, &model_dir, &model, language, device, repeat) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
    }
}

fn transcribe_file(
    path: &Path,
    model_dir: &Path,
    model: &str,
    language: Option<String>,
    device: Device,
    repeat: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = parse_model(model).ok_or("unknown model")?;
    let (pcm, rate) = decode_to_mono_f32(path)?;
    let choice = match device {
        Device::Auto => default_device(),
        Device::Cpu => BackendChoice::Cpu(cpu_device()),
    };

    let t = Instant::now();
    let engine = Engine::load_on(size, model_dir, choice)?;
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

fn record_test(secs: u64, device: Option<String>) -> Result<(), String> {
    let mut rec = CpalRecorder::new(device, 120);
    rec.start()?;
    std::thread::sleep(std::time::Duration::from_secs(secs));
    let r = rec.stop()?;
    println!("samples={}", r.samples.len());
    println!("rate={}", r.sample_rate);
    println!("duration_ms={}", r.duration_ms());
    println!("peak={:.4}", r.peak());
    Ok(())
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config: Config = Config::load_from(&config::default_path())?;
    config.validate()?;
    let rt = spawn(
        config,
        Box::new(|s| eprintln!("status: {s:?}")),
        Box::new(|m| eprintln!("notice: {m}")),
    );
    for line in std::io::stdin().lines() {
        match line?.trim() {
            "s" => rt.send(Input::Press),
            "e" => rt.send(Input::Release),
            "q" => break,
            _ => {}
        }
    }
    rt.shutdown();
    Ok(())
}
