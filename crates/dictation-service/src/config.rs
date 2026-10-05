use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use whisper_burn::ModelSize;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// Models the service can run. `ivrit-turbo` is the bundled Hebrew model
/// (ivrit-ai/whisper-large-v3-turbo); it shares `LargeV3Turbo` dimensions.
pub fn parse_model(name: &str) -> Option<ModelSize> {
    match name {
        "ivrit-turbo" => Some(ModelSize::LargeV3Turbo),
        other => ModelSize::parse(other).ok(),
    }
}

/// Float precision of the GPU model. f16 halves VRAM and is faster; the
/// service falls back to f32 when the GPU lacks f16 support.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    F16,
    F32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hotkey: String,
    pub model: String,
    pub model_dir: Option<PathBuf>,
    pub language: String,
    pub precision: Precision,
    pub input_device: Option<String>,
    pub paste_delay_ms: u64,
    pub max_record_secs: u32,
    pub queue_cap: usize,
    /// Type text while speaking (tap the hotkey to start and again to stop).
    pub stream: bool,
    pub stream_window_secs: u32,
    pub stream_interval_ms: u32,
    pub stream_overlap_ms: u32,
    /// Words closer than this to the end of the transcript may still be corrected.
    pub stream_lock_words: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "ctrl+space".into(),
            model: "ivrit-turbo".into(),
            model_dir: None,
            language: "he".into(),
            precision: Precision::F16,
            input_device: None,
            paste_delay_ms: 50,
            max_record_secs: 120,
            queue_cap: 3,
            stream: false,
            stream_window_secs: 3,
            stream_interval_ms: 1000,
            stream_overlap_ms: 400,
            stream_lock_words: 2,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| Err(ConfigError::Invalid(m.into()));
        if self.hotkey.trim().is_empty() {
            return bad("hotkey is empty");
        }
        if parse_model(&self.model).is_none() {
            return bad("unknown model");
        }
        let lang_ok = self.language == "auto"
            || (self.language.len() == 2 && self.language.chars().all(|c| c.is_ascii_lowercase()));
        if !lang_ok {
            return bad("language must be \"auto\" or a 2-letter lowercase code");
        }
        if self.queue_cap == 0 {
            return bad("queue_cap must be >= 1");
        }
        if self.max_record_secs == 0 {
            return bad("max_record_secs must be >= 1");
        }
        if !(1..=30).contains(&self.stream_window_secs) {
            return bad("stream_window_secs must be between 1 and 30");
        }
        if self.stream_interval_ms < 100 {
            return bad("stream_interval_ms must be >= 100");
        }
        if self.stream_overlap_ms >= self.stream_window_secs * 1000 {
            return bad("stream_overlap_ms must be smaller than the window");
        }
        if self.stream_lock_words == 0 {
            return bad("stream_lock_words must be >= 1");
        }
        Ok(())
    }

    /// `"auto"` means let whisper detect the language.
    pub fn language_option(&self) -> Option<String> {
        (self.language != "auto").then(|| self.language.clone())
    }

    /// A missing file yields the defaults; a present but invalid file is an error.
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(ConfigError::Io(e.to_string())),
        };
        let cfg: Self = toml::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        self.validate()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| ConfigError::Io(e.to_string()))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| ConfigError::Parse(e.to_string()))?;
        std::fs::write(path, text).map_err(|e| ConfigError::Io(e.to_string()))
    }
}

pub fn default_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dictation")
        .join("config.toml")
}
