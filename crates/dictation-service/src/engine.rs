use std::path::{Path, PathBuf};

use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::Wgpu;
use whisper_burn::backends::{BackendChoice, default_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::{ModelSize, Result, TranscriptionOptions, TranscriptionSegment, transcribe};

#[allow(clippy::large_enum_variant)]
enum Loaded {
    Cpu(Whisper<NdArray<f32>>),
    Wgpu(Whisper<Wgpu>),
}

/// A loaded model. Loading is strictly offline: weights come from a local
/// directory and nothing is ever downloaded.
pub struct Engine {
    model: Loaded,
}

/// Segments are joined with single spaces; whisper segments carry leading spaces.
pub fn join_segments(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .flat_map(|s| s.text.split_whitespace())
        .collect::<Vec<_>>()
        .join(" ")
}

/// First directory that contains `model.safetensors`: the configured one,
/// then `<exe_dir>/models/<model>`, then `<data_dir>/models/<model>`, then (Linux) `/usr/share/dictation/models/<model>`.
pub fn resolve_model_dir(
    model: &str,
    configured: Option<&Path>,
    exe_dir: &Path,
    data_dir: &Path,
) -> Option<PathBuf> {
    let system =
        cfg!(target_os = "linux").then(|| Path::new("/usr/share/dictation/models").join(model));
    configured
        .map(Path::to_path_buf)
        .into_iter()
        .chain([
            exe_dir.join("models").join(model),
            data_dir.join("models").join(model),
        ])
        .chain(system)
        .find(|d| d.join("model.safetensors").is_file())
}
impl Engine {
    pub fn load(size: ModelSize, model_dir: &Path) -> Result<Self> {
        Self::load_on(size, model_dir, default_device())
    }

    pub fn load_on(size: ModelSize, model_dir: &Path, choice: BackendChoice) -> Result<Self> {
        let model = match choice {
            BackendChoice::Cpu(d) => Loaded::Cpu(Whisper::load(size, model_dir, d)?),
            BackendChoice::Wgpu(d) => Loaded::Wgpu(Whisper::load(size, model_dir, d)?),
        };
        Ok(Self { model })
    }

    pub fn backend_name(&self) -> &'static str {
        match self.model {
            Loaded::Cpu(_) => "cpu",
            Loaded::Wgpu(_) => "wgpu",
        }
    }

    pub fn transcribe_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let options = TranscriptionOptions {
            language: language.map(str::to_owned),
            ..Default::default()
        };
        let result = match &self.model {
            Loaded::Cpu(m) => transcribe(m, pcm, sample_rate, &options)?,
            Loaded::Wgpu(m) => transcribe(m, pcm, sample_rate, &options)?,
        };
        Ok(join_segments(&result.segments))
    }
}
