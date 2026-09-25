use whisper_burn::config::ModelSize;
use whisper_burn::tokenizer::tiktoken::CoreBpe;
use whisper_burn::tokenizer::whisper::{STARTOPREV, Task, TextTokenizer};
use whisper_burn::transcribe::{
    TranscriptionOptions, chunk_prompt, push_untimestamped_fallback, validate_model_task,
    validate_options,
};

const MINI: &str = include_str!("fixtures/mini.tiktoken");

fn mini_tokenizer() -> TextTokenizer {
    let mut specials = vec![
        ("<|endoftext|>".to_string(), 50257),
        ("<|startoftranscript|>".to_string(), 50258),
        ("<|translate|>".to_string(), 50358),
        ("<|transcribe|>".to_string(), 50359),
        ("<|startoflm|>".to_string(), 50360),
        ("<|startofprev|>".to_string(), 50361),
        ("<|nospeech|>".to_string(), 50362),
        ("<|notimestamps|>".to_string(), 50363),
        ("<|0.00|>".to_string(), 50364),
    ];
    for (i, code) in TextTokenizer::LANGUAGES.iter().enumerate() {
        specials.push((format!("<|{code}|>"), 50259 + i as u32));
    }
    let core = CoreBpe::from_tiktoken_with_special(MINI.as_bytes(), &specials).unwrap();
    TextTokenizer::new_from_core(core, 51865, 1500).unwrap()
}

#[test]
fn validation_rejects_nonzero_temperature() {
    let ok = TranscriptionOptions {
        ..Default::default()
    };
    assert!(validate_options(&ok).is_ok());

    let bad = TranscriptionOptions {
        temperature: 0.25,
        ..Default::default()
    };
    assert!(validate_options(&bad).is_err());
    let bad = TranscriptionOptions {
        temperature: -1.0,
        ..Default::default()
    };
    assert!(validate_options(&bad).is_err());
}

#[test]
fn prompt_without_context_is_sot_lang_task() {
    let tok = mini_tokenizer();
    // [sot, <|en|>, <|transcribe|>]
    let p = chunk_prompt(&tok, &[], "en", Task::Transcribe, 448).unwrap();
    assert_eq!(p, vec![50258, 50259, 50359]);

    // translate task carries <|translate|>; language slots stay fixed
    let p = chunk_prompt(&tok, &[], "en", Task::Translate, 448).unwrap();
    assert_eq!(p, vec![50258, 50259, 50358]);
}

#[test]
fn prompt_prepends_previous_context_with_sot_prev() {
    let tok = mini_tokenizer();
    // [<|startofprev|>, prev..., <|sot|>, <|en|>, <|transcribe|>]
    let p = chunk_prompt(&tok, &[7, 5, 3], "en", Task::Transcribe, 448).unwrap();
    assert_eq!(p, vec![50361, 7, 5, 3, 50258, 50259, 50359]);
}

#[test]
fn prompt_truncates_context_to_n_ctx_half_minus_one() {
    let tok = mini_tokenizer();
    let context = vec![5u32; 300];
    let p = chunk_prompt(&tok, &context, "en", Task::Transcribe, 448).unwrap();
    // [sot_prev] + tail(223) + [sot, lang, task]
    assert_eq!(p.len(), 1 + 223 + 3);
    assert_eq!(p[0], STARTOPREV);
    assert!(p[1..224].iter().all(|&t| t == 5));
    assert_eq!(&p[1..224], &context[context.len() - 223..]);
    assert_eq!(&p[224..], &[50258, 50259, 50359]);
}

#[test]
fn fallback_emits_whole_chunk_segment_for_notimestamps_run() {
    let tok = mini_tokenizer();
    // prefix [sot, lang, task] + <|notimestamps|> + text + eot
    let prefix = vec![50258, 50259, 50359];
    let mut tokens = prefix.clone();
    tokens.extend([50363, 3, 5, 4, 50257]);
    let mut segs = Vec::new();
    push_untimestamped_fallback(&mut segs, &tokens, prefix.len(), &tok, 60_000).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 60_000);
    assert_eq!(segs[0].end, 90_000);
    assert!(
        !segs[0].text.contains("<|"),
        "specials leaked: {}",
        segs[0].text
    );
    assert!(segs[0].text.contains("hello"), "{}", segs[0].text);
}

#[test]
fn fallback_skips_when_no_text_tokens() {
    let tok = mini_tokenizer();
    let mut tokens = vec![50258, 50259, 50359];
    tokens.extend([50363, 50257]);
    let mut segs = Vec::new();
    push_untimestamped_fallback(&mut segs, &tokens, 3, &tok, 0).unwrap();
    assert!(segs.is_empty());
}

#[test]
fn turbo_translate_is_rejected_but_transcribe_and_large_v3_are_ok() {
    assert!(validate_model_task(ModelSize::LargeV3Turbo, Task::Translate).is_err());
    assert!(validate_model_task(ModelSize::LargeV3Turbo, Task::Transcribe).is_ok());
    assert!(validate_model_task(ModelSize::LargeV3, Task::Translate).is_ok());
    assert!(validate_model_task(ModelSize::IvritHebrew, Task::Translate).is_ok());
}

#[test]
fn specials_and_timestamps_are_not_text() {
    let tok = mini_tokenizer();
    assert!(tok.is_special(tok.sot()));
    assert!(tok.is_special(tok.eot()));
    assert!(tok.is_special(tok.transcribe()));
    assert!(tok.is_special(tok.notimestamps()));
    assert!(tok.is_special(tok.language_token("en").unwrap()));
    assert!(!tok.is_special(400));
    assert!(!tok.is_special(tok.timestamp_begin));
    assert!(tok.is_timestamp(tok.timestamp_begin));
}

#[cfg(all(feature = "ndarray", feature = "audio", feature = "weights"))]
mod network {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};
    use whisper_burn::audio::decode::decode_to_mono_f32;
    use whisper_burn::audio::resample::resample_to_16k;
    use whisper_burn::config::ModelSize;
    use whisper_burn::download::download_to;
    use whisper_burn::model::whisper::Whisper;
    use whisper_burn::transcribe::{ProgressUpdate, StageKind, transcribe};

    const JFK_URL: &str = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac";

    fn jfk_pcm() -> Vec<f32> {
        let dir = std::env::temp_dir().join("whisper-burn-t25");
        let flac = download_to(JFK_URL, &dir.join("jfk.flac"), false).unwrap();
        let (pcm, sr) = decode_to_mono_f32(&flac).unwrap();
        resample_to_16k(&pcm, sr, 16000).unwrap()
    }

    #[test]
    #[ignore = "network"]
    fn transcribe_english_speech_yields_segments() {
        let dev = NdArrayDevice::default();
        let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
        let pcm = jfk_pcm();

        let options = TranscriptionOptions {
            language: Some("en".to_string()),
            beam_size: 0,
            ..Default::default()
        };
        let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
        assert!(!segs.is_empty(), "no segments produced");
        assert_eq!(segs[0].start, 0, "first segment must start at 0");
        assert!(
            !segs[0].text.trim().is_empty(),
            "first segment text empty: {segs:?}"
        );
        let text: String = segs
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            text.to_lowercase().contains("country"),
            "expected the JFK quote, got: {text:?}"
        );
    }

    #[test]
    #[ignore = "network"]
    fn translate_english_speech_returns_translation_text() {
        let dev = NdArrayDevice::default();
        let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
        let pcm = jfk_pcm();

        let options = TranscriptionOptions {
            language: Some("en".to_string()),
            task: Task::Translate,
            beam_size: 0,
            ..Default::default()
        };
        let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
        assert!(!segs.is_empty(), "no segments produced");
        let text: String = segs
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!text.is_empty(), "empty translation: {segs:?}");
        assert!(
            !text.contains('<') && text.trim().len() >= 3,
            "expected real translation text, got: {text:?}"
        );
    }

    #[test]
    #[ignore = "network"]
    fn transcribe_streams_decode_segments_and_done_in_order() {
        let dev = NdArrayDevice::default();
        let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
        let pcm = jfk_pcm();
        let events = Arc::new(Mutex::new(Vec::<ProgressUpdate>::new()));
        let sink = events.clone();
        let cb: Arc<dyn Fn(&ProgressUpdate) + Send + Sync> =
            Arc::new(move |u| sink.lock().unwrap().push(u.clone()));
        let options = TranscriptionOptions {
            language: Some("en".to_string()),
            beam_size: 0,
            progress: Some(cb),
            ..Default::default()
        };
        let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
        assert!(!segs.is_empty(), "no segments produced");

        let events = events.lock().unwrap();
        assert!(
            matches!(events.first(), Some(ProgressUpdate::Stage { kind: StageKind::Decode, .. })),
            "first event should be a Decode stage, got {events:?}"
        );
        for seg in &segs {
            assert!(
                events.iter().any(|p| matches!(p, ProgressUpdate::Segment(s) if s == seg)),
                "segment streamed for {seg:?}"
            );
        }
        assert!(
            matches!(events.last(), Some(ProgressUpdate::Stage { kind: StageKind::Done, fraction: Some(1.0), .. })),
            "last event should be Done with fraction 1.0, got {events:?}"
        );
    }

    #[test]
    #[ignore = "network"]
    fn cancel_before_run_returns_empty_and_emits_done() {
        let dev = NdArrayDevice::default();
        let w = Whisper::<NdArray<f32>>::from_pretrained(ModelSize::Tiny, dev).unwrap();
        let pcm = jfk_pcm();
        let events = Arc::new(Mutex::new(Vec::<ProgressUpdate>::new()));
        let sink = events.clone();
        let cb: Arc<dyn Fn(&ProgressUpdate) + Send + Sync> =
            Arc::new(move |u| sink.lock().unwrap().push(u.clone()));
        let cancelled = Arc::new(AtomicBool::new(true));
        let options = TranscriptionOptions {
            language: Some("en".to_string()),
            beam_size: 0,
            progress: Some(cb),
            cancelled,
            ..Default::default()
        };
        let segs = transcribe(&w, &pcm, 16000, &options).unwrap();
        assert!(segs.is_empty(), "expected no segments, got {segs:?}");

        let events = events.lock().unwrap();
        assert!(
            !events.iter().any(|p| matches!(p, ProgressUpdate::Stage { kind: StageKind::Decode, .. })),
            "cancelled run must not start decoding, got {events:?}"
        );
        assert!(
            matches!(events.last(), Some(ProgressUpdate::Stage { kind: StageKind::Done, .. })),
            "cancelled run still emits Done, got {events:?}"
        );
    }
}
