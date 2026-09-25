use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_layer: usize,
    pub n_text_layer: usize,
    pub n_audio_state: usize,
    pub n_text_state: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub n_audio_ctx: usize,
    pub n_text_ctx: usize,
}

#[derive(Deserialize)]
struct HfConfig {
    #[serde(rename = "num_mel_bins")]
    n_mels: usize,
    #[serde(rename = "encoder_layers")]
    n_audio_layer: usize,
    #[serde(rename = "decoder_layers")]
    n_text_layer: usize,
    #[serde(rename = "d_model")]
    d_model: usize,
    #[serde(rename = "encoder_attention_heads")]
    n_head: usize,
    #[serde(rename = "vocab_size")]
    n_vocab: usize,
    #[serde(rename = "max_source_positions")]
    n_audio_ctx: usize,
    #[serde(rename = "max_target_positions")]
    n_text_ctx: usize,
}

impl ModelDimensions {
    pub fn from_config_json(s: &str) -> Result<Self> {
        let c: HfConfig =
            serde_json::from_str(s).map_err(|e| Error::ConfigParse(format!("config.json: {e}")))?;
        Ok(Self {
            n_mels: c.n_mels,
            n_audio_layer: c.n_audio_layer,
            n_text_layer: c.n_text_layer,
            n_audio_state: c.d_model,
            n_text_state: c.d_model,
            n_head: c.n_head,
            n_vocab: c.n_vocab,
            n_audio_ctx: c.n_audio_ctx,
            n_text_ctx: c.n_text_ctx,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    /// Hebrew-specialized fine-tune of `openai/whisper-large-v3` published by
    /// ivrit-ai; same dimensions/tokenizer as large-v3, released as
    /// sharded safetensors.
    IvritHebrew,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub n_mels: usize,
    pub n_vocab: usize,
    pub n_audio_layer: usize,
    pub n_text_layer: usize,
}

impl ModelSize {
    pub fn repo_id(self) -> &'static str {
        match self {
            ModelSize::Tiny => "openai/whisper-tiny",
            ModelSize::Base => "openai/whisper-base",
            ModelSize::Small => "openai/whisper-small",
            ModelSize::Medium => "openai/whisper-medium",
            ModelSize::Large => "openai/whisper-large",
            ModelSize::LargeV2 => "openai/whisper-large-v2",
            ModelSize::LargeV3 => "openai/whisper-large-v3",
            ModelSize::LargeV3Turbo => "openai/whisper-large-v3-turbo",
            ModelSize::IvritHebrew => "ivrit-ai/whisper-large-v3",
        }
    }

    /// True when the checkpoint is published as several safetensors shards
    /// (plus a `model.safetensors.index.json`) and must be merged at fetch
    /// time into a single `model.safetensors`.
    pub fn is_sharded(self) -> bool {
        matches!(self, ModelSize::IvritHebrew)
    }

    pub fn expected_dims(self) -> Dims {
        match self {
            ModelSize::Tiny => Dims {
                n_mels: 80,
                n_vocab: 51865,
                n_audio_layer: 4,
                n_text_layer: 4,
            },
            ModelSize::Base => Dims {
                n_mels: 80,
                n_vocab: 51865,
                n_audio_layer: 6,
                n_text_layer: 6,
            },
            ModelSize::Small => Dims {
                n_mels: 80,
                n_vocab: 51865,
                n_audio_layer: 12,
                n_text_layer: 12,
            },
            ModelSize::Medium => Dims {
                n_mels: 80,
                n_vocab: 51865,
                n_audio_layer: 24,
                n_text_layer: 24,
            },
            ModelSize::Large | ModelSize::LargeV2 => Dims {
                n_mels: 80,
                n_vocab: 51865,
                n_audio_layer: 32,
                n_text_layer: 32,
            },
            ModelSize::LargeV3 => Dims {
                n_mels: 128,
                n_vocab: 51866,
                n_audio_layer: 32,
                n_text_layer: 32,
            },
            ModelSize::LargeV3Turbo => Dims {
                n_mels: 128,
                n_vocab: 51866,
                n_audio_layer: 32,
                n_text_layer: 4,
            },
            ModelSize::IvritHebrew => ModelSize::LargeV3.expected_dims(),
        }
    }

    pub fn validate(&self, d: &ModelDimensions) -> Result<()> {
        let e = self.expected_dims();
        if d.n_mels != e.n_mels
            || d.n_vocab != e.n_vocab
            || d.n_audio_layer != e.n_audio_layer
            || d.n_text_layer != e.n_text_layer
        {
            return Err(Error::ConfigParse(format!(
                "{} config.json dims do not match expected {:?}",
                self.repo_id(),
                e
            )));
        }
        Ok(())
    }

    /// Every model, in CLI help order (also the dropdown order in the app).
    pub const ALL: [ModelSize; 9] = [
        ModelSize::Tiny,
        ModelSize::Base,
        ModelSize::Small,
        ModelSize::Medium,
        ModelSize::Large,
        ModelSize::LargeV2,
        ModelSize::LargeV3,
        ModelSize::LargeV3Turbo,
        ModelSize::IvritHebrew,
    ];

    /// The `--model`/dropdown key for this model.
    pub fn cli_name(self) -> &'static str {
        match self {
            ModelSize::Tiny => "tiny",
            ModelSize::Base => "base",
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
            ModelSize::LargeV2 => "large-v2",
            ModelSize::LargeV3 => "large-v3",
            ModelSize::LargeV3Turbo => "large-v3-turbo",
            ModelSize::IvritHebrew => "ivrit-hebrew",
        }
    }

    /// Parse a CLI/dropdown model key.
    pub fn parse(s: &str) -> Result<ModelSize> {
        Self::ALL
            .iter()
            .find(|m| m.cli_name() == s)
            .copied()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "unknown model {s:?}; expected tiny, base, small, medium, large, \
                     large-v2, large-v3, large-v3-turbo, or ivrit-hebrew"
                ))
            })
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelSize::Tiny => "Tiny",
            ModelSize::Base => "Base",
            ModelSize::Small => "Small",
            ModelSize::Medium => "Medium",
            ModelSize::Large => "Large",
            ModelSize::LargeV2 => "LargeV2",
            ModelSize::LargeV3 => "LargeV3",
            ModelSize::LargeV3Turbo => "LargeV3Turbo",
            ModelSize::IvritHebrew => "IvritHebrew",
        })
    }
}

/// Reads `config.json` at `path` and returns dimensions validated against
/// `size`'s expected shape.
pub fn read_config(path: &Path, size: ModelSize) -> Result<ModelDimensions> {
    let s = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let dims = ModelDimensions::from_config_json(&s)?;
    size.validate(&dims)?;
    Ok(dims)
}
