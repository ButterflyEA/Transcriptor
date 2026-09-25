#[cfg(feature = "audio")]
pub mod audio;

pub mod backends;
pub mod config;
pub mod decoding;
#[cfg(feature = "weights")]
pub mod download;
pub mod error;
pub mod features;
pub mod format;
pub mod model;
pub mod tokenizer;
pub mod transcribe;
pub mod weights;
pub use config::ModelSize;
pub use error::{Error, Result};
pub use transcribe::{TranscriptionOptions, TranscriptionSegment, transcribe, validate_options};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
