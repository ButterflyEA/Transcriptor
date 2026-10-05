use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::config::Precision;
use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::Wgpu;
use whisper_burn::backends::{BackendChoice, default_device};
use whisper_burn::model::whisper::Whisper;
use whisper_burn::{ModelSize, Result, TranscriptionOptions, TranscriptionSegment, transcribe};

#[allow(clippy::large_enum_variant)]
enum Loaded {
    Cpu(Whisper<NdArray<f32>>),
    Wgpu(Whisper<Wgpu>),
    WgpuF16(Whisper<Wgpu<burn::tensor::f16>>),
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
    /// Fixes the process-wide float precision. The burn device dtype can only
    /// be set before the first tensor is created, so call this before
    /// `default_device()`. The first call wins; f16 falls back to f32 when
    /// the GPU does not support it.
    pub fn resolve_precision(wanted: Precision) -> Precision {
        static CHOSEN: OnceLock<Precision> = OnceLock::new();
        *CHOSEN.get_or_init(|| {
            if wanted == Precision::F32 {
                return Precision::F32;
            }
            let set = catch_unwind(|| {
                burn::tensor::set_default_dtypes::<Wgpu<burn::tensor::f16>>(
                    &burn::backend::wgpu::WgpuDevice::DefaultDevice,
                    burn::tensor::DType::F16,
                    burn::tensor::DType::I32,
                )
            });
            match set {
                Ok(Ok(())) => Precision::F16,
                _ => Precision::F32,
            }
        })
    }

    pub fn load(size: ModelSize, model_dir: &Path, precision: Precision) -> Result<Self> {
        let precision = Self::resolve_precision(precision);
        Self::load_on(size, model_dir, default_device(), precision)
    }

    /// `precision` must come from `resolve_precision`, called before the device was probed.
    pub fn load_on(
        size: ModelSize,
        model_dir: &Path,
        choice: BackendChoice,
        precision: Precision,
    ) -> Result<Self> {
        let model = match choice {
            BackendChoice::Cpu(d) => Loaded::Cpu(Whisper::load(size, model_dir, d)?),
            BackendChoice::Wgpu(d) if precision == Precision::F16 => {
                Loaded::WgpuF16(Whisper::load(size, model_dir, d)?)
            }
            BackendChoice::Wgpu(d) => Loaded::Wgpu(Whisper::load(size, model_dir, d)?),
        };
        Ok(Self { model })
    }
    pub fn backend_name(&self) -> &'static str {
        match self.model {
            Loaded::Cpu(_) => "cpu",
            Loaded::Wgpu(_) => "wgpu",
            Loaded::WgpuF16(_) => "wgpu-f16",
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
            Loaded::WgpuF16(m) => transcribe(m, pcm, sample_rate, &options)?,
        };
        Ok(join_segments(&result.segments))
    }
}
