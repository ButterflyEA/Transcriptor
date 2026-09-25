//! `whisper-burn` CLI: transcribe audio with OpenAI Whisper on a burn
//! backend (T26).
//!
//! Usage: `whisper-burn [OPTIONS] <AUDIO>` — see `--help`.

use std::path::PathBuf;
use std::process::ExitCode;

use burn::tensor::backend::Backend;
use clap::{Parser, ValueEnum};

use whisper_burn::audio::decode::decode_to_mono_f32;
use whisper_burn::backends::{BackendChoice, cpu_device, try_wgpu_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::tokenizer::whisper::Task;
use whisper_burn::transcribe::validate_model_task;
use whisper_burn::{Error, ModelSize, Result, TranscriptionOptions, transcribe, validate_options};
use whisper_burn_cli::output;

#[derive(Parser)]
#[command(
    name = "whisper-burn",
    version,
    about = "Transcribe audio with OpenAI Whisper on a burn backend"
)]
struct Args {
    /// Path to an audio file (wav, flac, mp3, ...).
    #[arg(value_name = "AUDIO")]
    audio: PathBuf,

    /// Model size: tiny, base, small, medium, large, large-v2, large-v3,
    /// large-v3-turbo.
    #[arg(long, value_name = "SIZE", default_value = "tiny", value_parser = parse_model)]
    model: ModelSize,

    /// Spoken language code (e.g. "en"); auto-detected when omitted.
    #[arg(long)]
    language: Option<String>,

    /// Transcribe (same language) or translate to English.
    #[arg(long, value_enum, default_value_t = TaskArg::Transcribe)]
    task: TaskArg,

    /// Beam size; 0 selects greedy decoding.
    #[arg(long, value_name = "N", default_value_t = 0)]
    beam_size: usize,

    /// Print diagnostics to stderr.
    #[arg(long)]
    verbose: bool,

    /// Also write the transcript next to the audio in this format.
    #[arg(long, value_enum)]
    output: Option<OutputFormat>,

    /// Backend: wgpu (default, CPU fallback) or cpu.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Wgpu)]
    device: DeviceChoice,

    /// Stream stage progress (model, audio, per-window decode, segments)
    /// to stderr.
    #[arg(long)]
    progress: bool,

    /// Text prompt conditioning the first window (reference `initial_prompt`).
    #[arg(long)]
    initial_prompt: Option<String>,
}

fn parse_model(s: &str) -> std::result::Result<ModelSize, String> {
    ModelSize::parse(s).map_err(|e| e.to_string())
}

#[derive(Clone, Copy, ValueEnum)]
enum TaskArg {
    Transcribe,
    Translate,
}

impl From<TaskArg> for Task {
    fn from(task: TaskArg) -> Self {
        match task {
            TaskArg::Transcribe => Task::Transcribe,
            TaskArg::Translate => Task::Translate,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Txt,
    Srt,
    Vtt,
    Json,
}

impl OutputFormat {
    fn extension(self) -> &'static str {
        match self {
            OutputFormat::Txt => "txt",
            OutputFormat::Srt => "srt",
            OutputFormat::Vtt => "vtt",
            OutputFormat::Json => "json",
        }
    }

    fn render(self, segments: &[whisper_burn::TranscriptionSegment]) -> String {
        match self {
            OutputFormat::Txt => output::format_txt(segments),
            OutputFormat::Srt => output::format_srt(segments),
            OutputFormat::Vtt => output::format_vtt(segments),
            OutputFormat::Json => output::format_json(segments),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceChoice {
    Wgpu,
    Cpu,
}

fn pick_device(device: DeviceChoice) -> BackendChoice {
    match device {
        DeviceChoice::Cpu => BackendChoice::Cpu(cpu_device()),
        DeviceChoice::Wgpu => match try_wgpu_device() {
            Some(device) => BackendChoice::Wgpu(device),
            None => {
                eprintln!("no usable wgpu adapter; falling back to the CPU backend");
                BackendChoice::Cpu(cpu_device())
            }
        },
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.progress {
        install_progress_logger();
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<()> {
    validate_options(&TranscriptionOptions {
        language: args.language.clone(),
        task: args.task.into(),
        beam_size: args.beam_size,
        ..Default::default()
    })?;
    validate_model_task(args.model, args.task.into())?;

    let choice = pick_device(args.device);
    if args.verbose {
        eprintln!("model: {}", args.model.repo_id());
        eprintln!("device: {choice:?}");
        eprintln!("audio: {}", args.audio.display());
    }

    let (pcm, sample_rate) = decode_to_mono_f32(&args.audio)?;
    let name = args
        .audio
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.audio.display().to_string());
    log::info!(
        "audio: {name} ({:.1} s @ {} Hz, mono)",
        pcm.len() as f64 / sample_rate as f64,
        sample_rate
    );
    if args.verbose {
        eprintln!("decoded {} samples @ {sample_rate} Hz", pcm.len());
    }

    let segments = match choice {
        BackendChoice::Cpu(device) => transcribe_dev::<burn::backend::ndarray::NdArray<f32>>(
            &device,
            &pcm,
            sample_rate,
            args,
        )?,
        BackendChoice::Wgpu(device) => {
            transcribe_dev::<burn::backend::wgpu::Wgpu>(&device, &pcm, sample_rate, args)?
        }
    };

    let stdout = segments
        .iter()
        .map(output::bracket_line)
        .collect::<Vec<_>>()
        .join("\n");
    println!("{stdout}");

    if let Some(format) = args.output {
        let path = args.audio.with_extension(format.extension());
        let rendered = format.render(&segments);
        std::fs::write(&path, rendered).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        log::info!("wrote: {}", path.display());
        if args.verbose {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

/// Minimal stderr logger backing `--progress`: `[elapsed] message`.
struct ProgressLogger {
    start: std::time::Instant,
}

impl log::Log for ProgressLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info && !wants_suppressed(metadata.target())
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!(
                "[{:6.1}s] {}",
                self.start.elapsed().as_secs_f32(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

/// Keep the progress stream readable: ignore wgpu/burn backend internals,
/// which log through `log` too but only add noise for a transcript CLI.
fn wants_suppressed(target: &str) -> bool {
    target.starts_with("wgpu")
        || target.starts_with("burn_wgpu")
        || target.starts_with("cubecl_wgpu")
        || target.starts_with("burn::backend")
}

fn install_progress_logger() {
    let logger: &'static ProgressLogger = Box::leak(Box::new(ProgressLogger {
        start: std::time::Instant::now(),
    }));
    log::set_max_level(log::LevelFilter::Info);
    let _ = log::set_logger(logger);
}

fn transcribe_dev<B: Backend>(
    device: &B::Device,
    pcm: &[f32],
    sample_rate: u32,
    args: &Args,
) -> Result<Vec<whisper_burn::TranscriptionSegment>> {
    let model = Whisper::<B>::from_pretrained(args.model, device.clone())?;
    let options = TranscriptionOptions {
        language: args.language.clone(),
        task: args.task.into(),
        beam_size: args.beam_size,
        initial_prompt: args.initial_prompt.clone(),
        ..Default::default()
    };
    transcribe(&model, pcm, sample_rate, &options)
}
