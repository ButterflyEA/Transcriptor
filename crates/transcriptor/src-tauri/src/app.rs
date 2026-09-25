use serde::Serialize;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use whisper_burn::audio::decode::decode_to_mono_f32;
use whisper_burn::backends::{BackendChoice, cpu_device, try_wgpu_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::transcribe::{ProgressCallback, ProgressUpdate, StageKind, transcribe as burn_transcribe};
use whisper_burn::{ModelSize, TranscriptionSegment};

use crate::core::{
    DoneDto, ProgressDto, SegmentDto, TranscribeParams, parse_task, render, validate_params,
};
use crate::error::AppError;
use crate::state::AppState;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Defaults {
    pub models: Vec<String>,
    pub languages: Vec<String>,
    pub default_model: String,
    pub default_task: String,
    pub default_beam_size: usize,
    pub devices: Vec<String>,
}

#[tauri::command]
pub fn get_defaults() -> Result<Defaults, String> {
    let models = whisper_burn::config::ModelSize::ALL
        .iter()
        .map(|m| m.cli_name().to_string())
        .collect();
    let languages = {
        let mut v: Vec<String> = whisper_burn::decoding::lang::language_window(51866)
            .into_iter()
            .map(|(_, code)| code.to_string())
            .collect();
        v.insert(0, "auto".to_string());
        v
    };
    Ok(Defaults {
        models,
        languages,
        default_model: "tiny".to_string(),
        default_task: "transcribe".to_string(),
        default_beam_size: 0,
        devices: vec!["wgpu".to_string(), "cpu".to_string()],
    })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioInfo {
    pub path: String,
    pub name: String,
    pub size_bytes: u64,
}

#[tauri::command]
pub fn inspect_audio(path: String) -> Result<AudioInfo, AppError> {
    let meta = std::fs::metadata(&path)
        .map_err(|e| AppError::new("io", format!("cannot read file: {e}")))?;
    if !meta.is_file() {
        return Err(AppError::new("io", "not a file"));
    }
    let name = Path::new(&path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.clone());
    Ok(AudioInfo { path, name, size_bytes: meta.len() })
}

#[tauri::command]
pub fn transcribe(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    path: String,
    params: TranscribeParams,
) -> Result<(), AppError> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err(AppError::new("busy", "a transcription is already running"));
    }
    state.cancel.store(false, Ordering::SeqCst);
    let state = Arc::clone(state.inner());
    std::thread::spawn(move || run_pipeline(app, &state, path, params));
    Ok(())
}

#[tauri::command]
pub fn stop(state: State<'_, Arc<AppState>>) -> Result<(), AppError> {
    state.cancel.store(true, Ordering::SeqCst);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveRequest {
    pub path: String,
    pub format: String,
    pub include_timestamps: bool,
}

#[tauri::command]
pub fn save_transcript(
    state: State<'_, Arc<AppState>>,
    request: SaveRequest,
) -> Result<std::path::PathBuf, AppError> {
    save_transcript_direct(state.inner(), &request)
}

fn save_transcript_direct(
    state: &AppState,
    request: &SaveRequest,
) -> Result<std::path::PathBuf, AppError> {
    let segments = state.snapshot();
    if segments.is_empty() {
        return Err(AppError::new("no-transcript", "nothing to save yet — transcribe first"));
    }
    let bytes = render(&request.format, request.include_timestamps, &segments)?;
    std::fs::write(&request.path, bytes)
        .map_err(|e| AppError::new("io", format!("cannot write file: {e}")))?;
    Ok(std::path::PathBuf::from(&request.path))
}

fn run_pipeline(app: AppHandle, state: &AppState, path: String, params: TranscribeParams) {
    let done = match core_run(&app, state, &path, &params) {
        Ok(segments) => finish_run(state, segments),
        Err(e) => DoneDto {
            status: "error".into(),
            segments: vec![],
            message: Some(e.message.clone()),
        },
    };
    state.running.store(false, Ordering::SeqCst);
    let _ = app.emit("done", done);
}

fn finish_run(state: &AppState, segments: Vec<TranscriptionSegment>) -> DoneDto {
    let cancelled = state.cancel.load(Ordering::SeqCst);
    *state.last_segments.lock().unwrap() = Some(segments.clone());
    DoneDto {
        status: if cancelled { "cancelled".into() } else { "finished".into() },
        segments: segments.iter().map(SegmentDto::from).collect(),
        message: if cancelled { Some("stopped by user".into()) } else { None },
    }
}

fn core_run(
    app: &AppHandle,
    state: &AppState,
    path: &str,
    params: &TranscribeParams,
) -> std::result::Result<Vec<TranscriptionSegment>, AppError> {
    let model = ModelSize::parse(&params.model)?;
    let task = parse_task(&params.task)?;
    validate_params(model, task)?;

    let choice = match params.device.as_str() {
        "cpu" => BackendChoice::Cpu(cpu_device()),
        _ => match try_wgpu_device() {
            Some(device) => BackendChoice::Wgpu(device),
            None => BackendChoice::Cpu(cpu_device()),
        },
    };

    let progress = make_progress(app);
    let options = params
        .clone()
        .into_options(Arc::clone(&state.cancel), progress)?;

    emit_stage(app, "audio", format!("decoding {path}"), None);
    let (pcm, sample_rate) = decode_to_mono_f32(path)?;
    emit_stage(app, "audio", format!("audio: {:.1} s @ {} Hz", pcm.len() as f64 / sample_rate as f64, sample_rate), None);

    match choice {
        BackendChoice::Cpu(device) => run_backend::<burn::backend::ndarray::NdArray<f32>>(
            app, model, &device, &pcm, sample_rate, &options,
        ),
        BackendChoice::Wgpu(device) => run_backend::<burn::backend::wgpu::Wgpu>(
            app, model, &device, &pcm, sample_rate, &options,
        ),
    }
}

fn run_backend<B: burn::tensor::backend::Backend>(
    app: &AppHandle,
    model: ModelSize,
    device: &B::Device,
    pcm: &[f32],
    sample_rate: u32,
    options: &whisper_burn::TranscriptionOptions,
) -> std::result::Result<Vec<TranscriptionSegment>, AppError> {
    emit_stage(app, "weights", format!("loading {model}"), None);
    let whisper = Whisper::<B>::from_pretrained_with_progress(
        model,
        device.clone(),
        Some(make_download_callback(app)),
    )
    .map_err(AppError::from)?;
    emit_stage(app, "weights", format!("{model} ready"), None);
    burn_transcribe(&whisper, pcm, sample_rate, options).map_err(AppError::from)
}

fn make_download_callback(app: &AppHandle) -> whisper_burn::download::DownloadCallback {
    let handle = app.clone();
    Arc::new(move |downloaded: u64, total: Option<u64>| {
        let fraction = total.map(|t| downloaded as f32 / t.max(1) as f32);
        let _ = handle.emit(
            "progress",
            ProgressDto {
                kind: "download".into(),
                message: format!("downloading checkpoint ({}/…)", downloaded / (1024 * 1024)),
                fraction,
            },
        );
    })
}

fn make_progress(app: &AppHandle) -> ProgressCallback {
    let handle = app.clone();
    Arc::new(move |update: &ProgressUpdate| match update {
        ProgressUpdate::Stage { kind, message, fraction } => {
            let kind = match kind {
                StageKind::Download => "download",
                StageKind::Weights => "weights",
                StageKind::Audio => "audio",
                StageKind::Decode => "decode",
                StageKind::Done => "done",
            };
            let _ = handle.emit(
                "progress",
                ProgressDto { kind: kind.to_string(), message: message.clone(), fraction: *fraction },
            );
        }
        ProgressUpdate::Segment(segment) => {
            let _ = handle.emit("segment", SegmentDto::from(segment));
        }
    })
}

fn emit_stage(app: &AppHandle, kind: &str, message: impl Into<String>, fraction: Option<f32>) {
    let _ = app.emit(
        "progress",
        ProgressDto { kind: kind.to_string(), message: message.into(), fraction },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_serialize_as_camel_case() {
        let d = Defaults {
            models: vec!["tiny".into()],
            languages: vec!["auto".into()],
            default_model: "tiny".into(),
            default_task: "transcribe".into(),
            default_beam_size: 0,
            devices: vec!["wgpu".into()],
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["defaultModel"], "tiny");
        assert_eq!(v["defaultTask"], "transcribe");
        assert_eq!(v["defaultBeamSize"], 0);
    }

    #[test]
    fn save_request_deserializes_camel_case_wire_shape() {
        let wire = serde_json::json!({
            "path": "C:/out/transcript.docx",
            "format": "docx",
            "includeTimestamps": true
        });
        let r: SaveRequest = serde_json::from_value(wire).expect("wire shape must deserialize");
        assert_eq!(r.path, "C:/out/transcript.docx");
        assert_eq!(r.format, "docx");
        assert!(r.include_timestamps);
    }

    #[test]
    fn audio_info_serializes_as_camel_case() {
        let a = AudioInfo { path: "p".into(), name: "n".into(), size_bytes: 42 };
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["sizeBytes"], 42);
    }

    #[test]
    fn finished_run_stores_segments_for_save() {
        let state = AppState::default();
        let segs = vec![
            TranscriptionSegment { start: 0, end: 2500, text: "hello".into() },
            TranscriptionSegment { start: 2500, end: 5000, text: "world".into() },
        ];
        let dto = finish_run(&state, segs);
        assert_eq!(dto.status, "finished");
        assert_eq!(state.snapshot().len(), 2);

        let request = SaveRequest {
            path: std::env::temp_dir().join("transcriptor_save_test.docx").to_string_lossy().into_owned(),
            format: "docx".into(),
            include_timestamps: true,
        };
        let out = save_transcript_direct(&state, &request).expect("save after completed run must succeed");
        assert!(std::fs::metadata(&out).is_ok());
        let _ = std::fs::remove_file(&out);
    }
}
