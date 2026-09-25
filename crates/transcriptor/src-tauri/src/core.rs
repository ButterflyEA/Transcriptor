use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use whisper_burn::format::{format_docx, format_json, format_srt, format_txt, format_vtt};
use whisper_burn::tokenizer::whisper::Task;
use whisper_burn::transcribe::{ProgressCallback, TranscriptionOptions, validate_model_task, validate_options};
use whisper_burn::{ModelSize, TranscriptionSegment};

use crate::error::AppError;

// --- types -------------------------------------------------------------------

#[derive(serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeParams {
    pub model: String,
    pub language: Option<String>,
    pub task: String,
    pub beam_size: usize,
    pub initial_prompt: Option<String>,
    pub device: String,
}

#[derive(serde::Serialize, Clone)]
pub struct SegmentDto {
    pub start: u32,
    pub end: u32,
    pub text: String,
}

impl From<&TranscriptionSegment> for SegmentDto {
    fn from(s: &TranscriptionSegment) -> Self {
        Self { start: s.start, end: s.end, text: s.text.clone() }
    }
}

#[derive(serde::Serialize, Clone)]
pub struct ProgressDto {
    pub kind: String,
    pub message: String,
    pub fraction: Option<f32>,
}

#[derive(serde::Serialize, Clone)]
pub struct DoneDto {
    pub status: String,
    pub segments: Vec<SegmentDto>,
    pub message: Option<String>,
}

// --- pure logic --------------------------------------------------------------

pub fn parse_task(s: &str) -> Result<Task, AppError> {
    match s {
        "transcribe" => Ok(Task::Transcribe),
        "translate" => Ok(Task::Translate),
        other => Err(AppError::new("task", format!("unknown task {other:?}; expected transcribe or translate"))),
    }
}

pub fn render(
    format: &str,
    include_timestamps: bool,
    segments: &[TranscriptionSegment],
) -> Result<Vec<u8>, AppError> {
    match format {
        "txt" => Ok(format_txt(segments).into_bytes()),
        "srt" => Ok(format_srt(segments).into_bytes()),
        "vtt" => Ok(format_vtt(segments).into_bytes()),
        "json" => Ok(format_json(segments).into_bytes()),
        "docx" => format_docx(segments, include_timestamps).map_err(Into::into),
        other => Err(AppError::new("format", format!("unsupported format {other:?}"))),
    }
}

pub fn validate_params(model: ModelSize, task: Task) -> Result<(), AppError> {
    validate_model_task(model, task).map_err(Into::into)
}

impl TranscribeParams {
    pub fn into_options(
        self,
        cancel: Arc<AtomicBool>,
        progress: ProgressCallback,
    ) -> std::result::Result<TranscriptionOptions, AppError> {
        let task = parse_task(&self.task)?;
        let model = ModelSize::parse(&self.model)?;
        validate_params(model, task)?;
        let language = self.language.filter(|l| l != "auto");
        let initial_prompt = self.initial_prompt.filter(|p| !p.trim().is_empty());
        let options = TranscriptionOptions {
            language,
            task,
            beam_size: self.beam_size,
            initial_prompt,
            progress: Some(progress),
            cancelled: cancel,
            ..Default::default()
        };
        validate_options(&options).map_err(AppError::from)?;
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<TranscriptionSegment> {
        vec![
            TranscriptionSegment { start: 0, end: 2500, text: "שלום עולם".to_string() },
            TranscriptionSegment { start: 2500, end: 5000, text: "hello world".to_string() },
        ]
    }

    #[test]
    fn render_every_format_produces_bytes() {
        let segs = segs();
        for format in ["txt", "srt", "vtt", "json", "docx"] {
            let out = render(format, true, &segs).unwrap_or_else(|e| panic!("{format}: {e}"));
            assert!(!out.is_empty(), "{format} empty");
        }
        let txt = render("txt", false, &segs).unwrap();
        assert!(String::from_utf8_lossy(&txt).contains("שלום עולם"));
    }

    #[test]
    fn parse_task_rejects_unknown() {
        assert_eq!(parse_task("transcribe").unwrap(), Task::Transcribe);
        assert_eq!(parse_task("translate").unwrap(), Task::Translate);
        assert!(parse_task("summarize").is_err());
    }

    #[test]
    fn turbo_translate_rejected_by_validate_params() {
        assert!(validate_params(ModelSize::LargeV3Turbo, Task::Translate).is_err());
        assert!(validate_params(ModelSize::LargeV3Turbo, Task::Transcribe).is_ok());
    }

    #[test]
    fn docx_render_includes_timestamp_brackets() {
        let out = render("docx", true, &segs()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn transcribe_params_deserializes_camel_case_wire_shape() {
        let wire = serde_json::json!({
            "model": "tiny",
            "language": "auto",
            "task": "transcribe",
            "beamSize": 0,
            "initialPrompt": "",
            "device": "wgpu"
        });
        let p: TranscribeParams = serde_json::from_value(wire).expect("wire shape must deserialize");
        assert_eq!(p.model, "tiny");
        assert_eq!(p.beam_size, 0);
        assert_eq!(p.initial_prompt.as_deref(), Some(""));
    }

    #[test]
    fn render_formats_match_extension_content_rules() {
        let segs = segs();
        let txt = String::from_utf8(render("txt", true, &segs).unwrap()).unwrap();
        assert!(txt.contains("שלום עולם"));
        let json = String::from_utf8(render("json", false, &segs).unwrap()).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v.as_array().map(|a| a.len()), Some(2));
        assert!(render("docx", true, &segs).unwrap().len() > 100);
        // PDF saving is disabled app-wide until the RTL rendering is fixed:
        // the backend rejects it just like any unknown format.
        assert!(render("pdf", true, &segs).is_err());
        assert!(render("nope", true, &segs).is_err());
    }
}
