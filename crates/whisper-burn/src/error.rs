use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("download failed: {0}")]
    Download(String),
    #[error("missing weights file: {path}")]
    MissingWeights { path: PathBuf },
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("config parse error: {0}")]
    ConfigParse(String),
    #[error("missing weight in checkpoint: {0}")]
    MissingWeight(String),
    #[error("unexpected extra weight in checkpoint: {0}")]
    UnexpectedWeight(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("decoder error: {0}")]
    Decoder(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("audio decode error: {0}")]
    AudioDecode(String),
    #[error("resample error: {0}")]
    Resample(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Error::Io {
            path: PathBuf::new(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_readable_errors() {
        let e = Error::ShapeMismatch {
            name: "enc.conv1.weight".into(),
            expected: vec![1, 2],
            got: vec![3, 4],
        };
        assert!(e.to_string().contains("enc.conv1.weight"));
        let e = Error::MissingWeight("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn io_error_converts() {
        let e: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(e, Error::Io { .. }));
    }
}
