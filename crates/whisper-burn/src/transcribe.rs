//! End-to-end transcription: raw PCM in, timestamped segments out.
//!
//! Mirrors `whisper.transcribe`'s orchestration: resample to 16 kHz (when
//! needed), split into 30 s zero-padded chunks (reference `split_chunks` /
//! `pad_or_trim`), then per chunk: mel → encoder → language detection (only
//! when not given) → greedy/beam decode → no-speech filtering → reference
//! segment assembly offset by the chunk's seek time. Cross-chunk context is
//! fed back through the reference `<|startofprev|>` prompt slot when
//! `condition_on_previous_text` is set, capped at `n_ctx / 2 - 1` tokens.
//!
//! v1 divergences from the reference (deliberate, tracked):
//! - `temperature` is pinned to `0.0` (`Err(Unsupported)` otherwise); there is
//!   no temperature fallback ladder.
//! - `best_of` is carried for API parity but unused (sampling path).
//! - Timestamp-driven seek advance is not implemented: chunks always advance
//!   by a full 30 s window (reference seeks to the last timestamp).
//! - Language is detected once on the first chunk (the reference also detects
//!   once, up to the first 30 s).
//! - Decoders run without the reference's timestamp rules, so the model may
//!   emit `<|notimestamps|>`; such runs have no timestamp boundaries and are
//!   assembled as one whole-chunk segment (start = seek, end = seek + 30 s).

#[cfg(feature = "audio")]
use crate::audio::resample::resample_to_16k;
use crate::decoding::beam::{BeamOptions, beam_search};
use crate::decoding::greedy::{GreedyOptions, greedy_search};
use crate::decoding::{assemble_segments, detect_language};
use crate::features::mel::FeatureExtractor;
use crate::features::window::{N_FRAMES, split_chunks};
use crate::model::whisper::Whisper;
use crate::tokenizer::whisper::{Task, TextTokenizer};
use crate::config::ModelSize;
use crate::{Error, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The segments `transcribe` returns: millisecond start/end plus decoded text.
pub use crate::decoding::segments::Segment as TranscriptionSegment;

/// Pipeline stages surfaced to UIs via the progress callback. `Download`
/// carries a completion `fraction` when the total size is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    Download,
    Weights,
    Audio,
    Decode,
    Done,
}

/// A live event from a running transcription pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgressUpdate {
    /// Stage transition with a human-readable `message`.
    Stage {
        kind: StageKind,
        message: String,
        fraction: Option<f32>,
    },
    /// One finished segment, streamed as it is decoded.
    Segment(TranscriptionSegment),
}

/// Shared progress sink; the desktop app forwards events to the UI.
pub type ProgressCallback = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;

/// Full-pipeline transcription options. Defaults match the reference whisper
/// CLI/API (`no_speech_threshold=0.6`, `logprob_threshold=-1.0`,
/// `condition_on_previous_text=True`, `best_of=5`).
#[derive(Clone)]
pub struct TranscriptionOptions {
    /// Spoken language code (e.g. `"en"`); `None` triggers one-shot detection
    /// on the first chunk.
    pub language: Option<String>,
    /// `Transcribe` (same language) or `Translate` (to English).
    pub task: Task,
    /// `0` → greedy; `>= 1` → beam search.
    pub beam_size: usize,
    /// Reference `best_of` (sampling at `temperature > 0`); unused by the
    /// deterministic path, carried for API parity.
    pub best_of: usize,
    /// v1 pins this to `0.0`; anything else is `Err(Unsupported)`.
    pub temperature: f32,
    /// Feed the previous window's output back as a `<|startofprev|>` prompt.
    pub condition_on_previous_text: bool,
    /// Skip chunks whose `<|nospeech|>` probability exceeds this (reference
    /// `no_speech_threshold`).
    pub no_speech_threshold: f32,
    /// A chunk with `avg_logprob` above this is kept even under a
    /// no-speech flag (reference `logprob_threshold`).
    pub logprob_threshold: f32,
    /// Optional text to prompt the first window with (reference
    /// `initial_prompt`, space-prefixed before encoding).
    pub initial_prompt: Option<String>,
    /// Stream stage/segment events as the pipeline runs; `None` disables
    /// streaming.
    pub progress: Option<ProgressCallback>,
    /// Best-effort cancellation: set true and the decoder stops cleanly
    /// after the current window, returning segments produced so far.
    pub cancelled: Arc<AtomicBool>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: Task::Transcribe,
            beam_size: 0,
            best_of: 5,
            temperature: 0.0,
            condition_on_previous_text: true,
            no_speech_threshold: 0.6,
            logprob_threshold: -1.0,
            initial_prompt: None,
            progress: None,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl std::fmt::Debug for TranscriptionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionOptions")
            .field("language", &self.language)
            .field("task", &self.task)
            .field("beam_size", &self.beam_size)
            .field("best_of", &self.best_of)
            .field("temperature", &self.temperature)
            .field("condition_on_previous_text", &self.condition_on_previous_text)
            .field("no_speech_threshold", &self.no_speech_threshold)
            .field("logprob_threshold", &self.logprob_threshold)
            .field("initial_prompt", &self.initial_prompt)
            .field("progress", &"<streaming callback>")
            .field("cancelled", &self.cancelled.load(Ordering::SeqCst))
            .finish()
    }
}

/// Rejects model/task combinations the engine cannot honor: whisper
/// large-v3-turbo was fine-tuned on transcription data only and re-emits the
/// source language under `--task translate` (OpenAI model card).
pub fn validate_model_task(model: ModelSize, task: Task) -> Result<()> {
    if matches!(model, ModelSize::LargeV3Turbo) && matches!(task, Task::Translate) {
        return Err(Error::Unsupported(
            "whisper-large-v3-turbo was not trained for translation, \
             only transcription; --task translate would re-transcribe the \
             source language. Use --task transcribe, or translate with a \
             multilingual model (--model large-v3, medium, small, base, tiny)."
                .to_string(),
        ));
    }
    Ok(())
}

/// Rejects option combinations v1 cannot honor (currently: any temperature
/// other than `0.0`).
pub fn validate_options(options: &TranscriptionOptions) -> Result<()> {
    if options.temperature != 0.0 {
        return Err(Error::Unsupported(format!(
            "temperature must be 0.0 in v1 (deterministic decode), got {}",
            options.temperature
        )));
    }
    Ok(())
}

/// Forward a progress event to the configured callback (no-op when absent).
fn emit(options: &TranscriptionOptions, update: &ProgressUpdate) {
    if let Some(callback) = &options.progress {
        callback(update);
    }
}

/// The decoder prompt for one window: `[<|startofprev|>] + context_tail +
/// [<|sot|>, <|language|>, <|task|>]`, with the context tail capped at
/// `n_ctx / 2 - 1` tokens (reference `_get_initial_tokens`).
pub fn chunk_prompt(
    tokenizer: &TextTokenizer,
    context: &[u32],
    language: &str,
    task: Task,
    n_ctx: usize,
) -> Result<Vec<u32>> {
    let cap = n_ctx / 2 - 1;
    let tail: &[u32] = if context.len() > cap {
        &context[context.len() - cap..]
    } else {
        context
    };
    let seq = tokenizer.sot_sequence(Some(language), task, true)?;
    log::info!(
        "prompt: language={} task={:?} tokens={:?}",
        language,
        task,
        seq
    );

    if !tail.is_empty() {
        let mut with_prev = Vec::with_capacity(tail.len() + seq.len() + 1);
        with_prev.push(tokenizer.startofprev());
        with_prev.extend_from_slice(tail);
        with_prev.extend_from_slice(&seq);
        return Ok(with_prev);
    }
    Ok(seq)
}

/// v1 fallback for chunks the decoders rendered without the reference's
/// timestamp rules: when nothing timestamp-bounded assembled but the window
/// produced real text tokens, emit a single whole-window segment (start = seek,
/// end = seek + 30 s) with the specials stripped from the decoded text.
pub fn push_untimestamped_fallback(
    segments: &mut Vec<TranscriptionSegment>,
    tokens: &[u32],
    prefix_len: usize,
    tokenizer: &TextTokenizer,
    seek_ms: u32,
) -> Result<()> {
    let text_tokens: Vec<u32> = tokens[prefix_len..]
        .iter()
        .copied()
        .filter(|&t| !tokenizer.is_timestamp(t) && !tokenizer.is_special(t))
        .collect();
    if text_tokens.is_empty() {
        return Ok(());
    }
    let text = tokenizer.decode(&text_tokens)?;
    if text.trim().is_empty() {
        return Ok(());
    }
    segments.push(TranscriptionSegment {
        start: seek_ms,
        end: seek_ms + 30_000,
        text,
    });
    Ok(())
}

/// Transcribe raw PCM into timestamped segments. `sample_rate` is the PCM's
/// rate; anything but 16 kHz is resampled (requires the `audio` feature).
pub fn transcribe<B: Backend>(
    whisper: &Whisper<B>,
    pcm: &[f32],
    sample_rate: u32,
    options: &TranscriptionOptions,
) -> Result<Vec<TranscriptionSegment>> {
    validate_options(options)?;
    let tokenizer = TextTokenizer::standard(whisper.dims.n_vocab as u32, whisper.dims.n_audio_ctx)?;

    let pcm16 = if sample_rate == 16_000 {
        pcm.to_vec()
    } else {
        let out: Vec<f32> = {
            #[cfg(feature = "audio")]
            {
                resample_to_16k(pcm, sample_rate, 16_000)?
            }
            #[cfg(not(feature = "audio"))]
            {
                return Err(Error::Unsupported(
                    "resampling non-16 kHz audio needs the `audio` feature".into(),
                ));
            }
        };
        log::info!(
            "audio: resampled {} Hz -> 16 kHz ({} samples)",
            sample_rate,
            out.len()
        );
        out
    };
    let chunks = split_chunks(&pcm16);
    if chunks.is_empty() {
        emit(
            options,
            &ProgressUpdate::Stage {
                kind: StageKind::Done,
                message: "no windows to decode".into(),
                fraction: None,
            },
        );
        return Ok(vec![]);
    }
    log::info!(
        "decode: {:.1} s of audio in {} window(s) of 30 s",
        pcm16.len() as f64 / 16_000.0,
        chunks.len()
    );

    let fe = FeatureExtractor::new(whisper.dims.n_mels, 16_000)?;
    let device = whisper
        .devices()
        .into_iter()
        .next()
        .ok_or_else(|| Error::Decoder("model registered no device".into()))?;

    // Cross-chunk context: the initial prompt goes first, then each placed
    // chunk's decoded run, mirroring the reference `all_tokens` accumulator.
    let mut context: Vec<u32> = Vec::new();
    if let Some(prompt) = &options.initial_prompt {
        let trimmed = format!(" {}", prompt.trim());
        if !trimmed.trim().is_empty() {
            context.extend(tokenizer.encode(&trimmed)?);
        }
    }
    let mut resolved_language: Option<String> = options.language.clone();

    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let windows = chunks.len();
    for (ci, (chunk, _)) in chunks.iter().enumerate() {
        let seek_ms = (ci * 30_000) as u32;
        if options.cancelled.load(Ordering::SeqCst) {
            log::info!("decode: window {}/{}: cancelled", ci + 1, windows);
            break;
        }
        emit(
            options,
            &ProgressUpdate::Stage {
                kind: StageKind::Decode,
                message: format!("decode window {}/{}", ci + 1, windows),
                fraction: Some((ci + 1) as f32 / windows as f32),
            },
        );
        log::info!(
            "decode: window {}/{} ({} – {})",
            ci + 1,
            windows,
            ms_clock(seek_ms),
            ms_clock(seek_ms + 30_000)
        );

        let mel = fe.log_mel(chunk)?;
        let xa = whisper.forward_encoder(Tensor::<B, 3>::from_data(
            TensorData::new(mel, [1, whisper.dims.n_mels, N_FRAMES]),
            &device,
        ));

        if resolved_language.is_none() {
            log::info!("decode: window {}/{}: detecting language", ci + 1, windows);
            resolved_language = Some(detect_language(whisper, &xa)?.language.to_string());
            log::info!(
                "decode: detected language {}",
                resolved_language.as_deref().unwrap_or("en")
            );
        }
        let language = resolved_language.as_deref().unwrap_or("en");

        let mut tokens = chunk_prompt(
            &tokenizer,
            &context,
            language,
            options.task,
            whisper.dims.n_text_ctx,
        )?;
        let prefix_len = tokens.len();

        let (no_speech_prob, mean_logprob) = if options.beam_size >= 1 {
            let beam = BeamOptions {
                beam_size: options.beam_size,
                best_of: options.best_of,
                ..Default::default()
            };
            let r = beam_search(whisper, &xa, &mut tokens, &beam)?;
            (r.no_speech_prob, r.mean_logprob)
        } else {
            let greedy = GreedyOptions::default();
            let r = greedy_search(whisper, &xa, &mut tokens, &greedy)?;
            (r.no_speech_prob, r.mean_logprob)
        };

        // Reference no-speech filter: skip when <|nospeech|> is likely AND the
        // decode did not rescue the chunk with a high average log-probability.
        if no_speech_prob > options.no_speech_threshold && mean_logprob <= options.logprob_threshold
        {
            log::info!(
                "decode: window {}/{}: no-speech ({:.3}), skipped",
                ci + 1,
                windows,
                no_speech_prob
            );
            continue;
        }
        log::info!(
            "decode: window {}/{}: language={} beam={} tokens={} no-speech={:.3} mean-logprob={:.3}",
            ci + 1,
            windows,
            language,
            options.beam_size,
            tokens.len().saturating_sub(prefix_len),
            no_speech_prob,
            mean_logprob
        );

        let from = segments.len();
        let assembled = assemble_segments(&tokens, &tokenizer, seek_ms)?;
        if assembled.is_empty() {
            push_untimestamped_fallback(&mut segments, &tokens, prefix_len, &tokenizer, seek_ms)?;
        } else {
            segments.extend(assembled);
        }
        for segment in &segments[from..] {
            log::info!("  {}", segment_progress(segment));
            emit(options, &ProgressUpdate::Segment(segment.clone()));
        }

        if options.condition_on_previous_text {
            context.extend(tokens[prefix_len..].iter().copied().filter(|&t| t != tokenizer.eot()));
        }
    }
    emit(
        options,
        &ProgressUpdate::Stage {
            kind: StageKind::Done,
            message: format!("{} segment(s)", segments.len()),
            fraction: Some(1.0),
        },
    );
    Ok(segments)
}

/// `MM:SS.mmm` clock used by progress logging.
fn ms_clock(ms: u32) -> String {
    let (ms, secs) = (ms % 1000, ms / 1000);
    let (secs, mins) = (secs % 60, secs / 60);
    format!("{mins:02}:{secs:02}.{ms:03}")
}

/// Console-style progress line for a finished segment.
fn segment_progress(segment: &TranscriptionSegment) -> String {
    format!(
        "[{} --> {}]  {}",
        ms_clock(segment.start),
        ms_clock(segment.end),
        segment.text
    )
}
