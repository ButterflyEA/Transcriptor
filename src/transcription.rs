use std::{io::Cursor, path::PathBuf};

use anyhow::{Context, Result, bail};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as whisper, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "openai/whisper-tiny.en";
const MODEL_REVISION: &str = "main";

pub fn transcribe_wav_bytes(audio_bytes: &[u8]) -> Result<String> {
    let (pcm_data, sample_rate) = decode_wav(audio_bytes)?;
    if sample_rate != whisper::SAMPLE_RATE as u32 {
        bail!(
            "unsupported sample rate: {sample_rate}. Expected {} Hz wav audio",
            whisper::SAMPLE_RATE
        );
    }

    let api = Api::new().context("failed to initialize Hugging Face API")?;
    let repo = Repo::with_revision(
        MODEL_ID.to_string(),
        RepoType::Model,
        MODEL_REVISION.to_string(),
    );
    let api_repo = api.repo(repo);

    let config_file = api_repo
        .get("config.json")
        .context("failed to download whisper config")?;
    let tokenizer_file = api_repo
        .get("tokenizer.json")
        .context("failed to download whisper tokenizer")?;
    let weights_file = api_repo
        .get("model.safetensors")
        .context("failed to download whisper weights")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    let mel_filters = decode_mel_filters(config.num_mel_bins)?;
    let mel = whisper::audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();

    let device = Device::Cpu;
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device,
    )?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[PathBuf::from(weights_file)], whisper::DTYPE, &device)
    }?;
    let model = whisper::model::Whisper::load(&vb, config)?;

    greedy_decode(model, tokenizer, mel)
}

fn decode_wav(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = Cursor::new(audio_bytes);
    let mut reader = hound::WavReader::new(cursor).context("invalid wav payload")?;
    let spec = reader.spec();

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read float samples")?,
        hound::SampleFormat::Int => {
            let max = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|value| value.map(|v| v as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read integer samples")?
        }
    };

    Ok((samples, spec.sample_rate))
}

fn decode_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
    let raw = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        _ => bail!("unsupported mel bin count: {num_mel_bins}"),
    };

    let mut mel_filters = vec![0f32; raw.len() / 4];
    LittleEndian::read_f32_into(raw, &mut mel_filters);
    Ok(mel_filters)
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .with_context(|| format!("missing required tokenizer token: {token}"))
}

fn greedy_decode(
    mut model: whisper::model::Whisper,
    tokenizer: Tokenizer,
    mel: Tensor,
) -> Result<String> {
    let sot_token = token_id(&tokenizer, whisper::SOT_TOKEN)?;
    let transcribe_token = token_id(&tokenizer, whisper::TRANSCRIBE_TOKEN)?;
    let no_timestamps_token = token_id(&tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
    let eot_token = token_id(&tokenizer, whisper::EOT_TOKEN)?;

    let audio_features = model.encoder.forward(&mel, true)?;
    let mut tokens = vec![sot_token, transcribe_token, no_timestamps_token];
    let sample_len = model.config.max_target_positions / 2;

    for i in 0..sample_len {
        let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
        let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

        let (_, seq_len, _) = ys.dims3()?;
        let logits = model
            .decoder
            .final_linear(&ys.i((..1, seq_len - 1..))?)?
            .i(0)?
            .i(0)?;

        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index as u32)
            .context("failed to select next token")?;

        tokens.push(next_token);

        if next_token == eot_token || tokens.len() > model.config.max_target_positions {
            break;
        }
    }

    tokenizer
        .decode(&tokens, true)
        .map(|text| text.trim().to_string())
        .map_err(anyhow::Error::msg)
}
