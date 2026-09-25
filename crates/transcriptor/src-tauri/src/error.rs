#[derive(Debug, serde::Serialize)]
pub struct AppError {
    pub kind: String,
    pub message: String,
}

impl AppError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self { kind: kind.into(), message: message.into() }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AppError {}

impl From<whisper_burn::Error> for AppError {
    fn from(e: whisper_burn::Error) -> Self {
        use whisper_burn::Error as W;
        let kind = match &e {
            W::Io { .. } => "io",
            W::Download(_) => "download",
            W::MissingWeights { .. } => "missing-weights",
            W::UnsupportedFormat(_) => "format",
            W::ConfigParse(_) => "config",
            W::MissingWeight(_) => "weights",
            W::UnexpectedWeight(_) => "weights",
            W::ShapeMismatch { .. } => "shape",
            W::Decoder(_) => "decode",
            W::Tokenizer(_) => "tokenizer",
            W::AudioDecode(_) => "audio",
            W::Resample(_) => "audio",
            W::Unsupported(_) => "unsupported",
        };
        Self { kind: kind.to_string(), message: e.to_string() }
    }
}