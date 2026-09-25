//! `whisper-burn` CLI library.
//!
//! The binary (`main.rs`) is deliberately thin; renderable logic lives here so
//! integration tests can exercise it (binary-only crates cannot be imported).

pub mod output;
