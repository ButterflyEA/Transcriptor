use crate::config::{ModelDimensions, ModelSize, read_config};
use crate::error::{Error, Result};
use crate::model::attention::MultiHeadAttention;
use crate::model::block::ResidualAttentionBlock;
use crate::model::decoder::Decoder;
use crate::model::encoder::{Conv1d, Encoder};
use crate::model::ops::{LayerNorm, Linear};
use crate::weights::loader::WeightMap;
use burn::module::{Module, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::path::Path;

fn ln<B: Backend>(wm: &mut WeightMap<B>, prefix: &str, n: usize) -> Result<LayerNorm<B>> {
    Ok(LayerNorm {
        weight: Param::from_tensor(wm.take_1d(&format!("{prefix}.weight"), n)?),
        bias: Param::from_tensor(wm.take_1d(&format!("{prefix}.bias"), n)?),
        epsilon: 1e-5,
    })
}

/// Like `ln` but tolerates a layer that exists in some checkpoint variants only.
fn ln_opt<B: Backend>(
    wm: &mut WeightMap<B>,
    prefix: &str,
    n: usize,
) -> Result<Option<LayerNorm<B>>> {
    Ok(match wm.take_opt_1d(&format!("{prefix}.weight"), n)? {
        Some(weight) => Some(LayerNorm {
            weight: Param::from_tensor(weight),
            bias: Param::from_tensor(wm.take_1d(&format!("{prefix}.bias"), n)?),
            epsilon: 1e-5,
        }),
        None => None,
    })
}

fn linear<B: Backend>(
    wm: &mut WeightMap<B>,
    prefix: &str,
    out: usize,
    nin: usize,
) -> Result<Linear<B>> {
    Ok(Linear {
        weight: Param::from_tensor(wm.take_2d(&format!("{prefix}.weight"), [out, nin])?),
        bias: Some(Param::from_tensor(
            wm.take_1d(&format!("{prefix}.bias"), out)?,
        )),
    })
}

/// `Linear` without a bias term (reference whisper's key projections).
fn linear_nobias<B: Backend>(
    wm: &mut WeightMap<B>,
    prefix: &str,
    out: usize,
    nin: usize,
) -> Result<Linear<B>> {
    Ok(Linear {
        weight: Param::from_tensor(wm.take_2d(&format!("{prefix}.weight"), [out, nin])?),
        bias: None,
    })
}

fn mha<B: Backend>(
    wm: &mut WeightMap<B>,
    prefix: &str,
    n_state: usize,
    n_head: usize,
) -> Result<MultiHeadAttention<B>> {
    let q = linear(wm, &format!("{prefix}.query"), n_state, n_state)?;
    let k = linear_nobias(wm, &format!("{prefix}.key"), n_state, n_state)?;
    let v = linear(wm, &format!("{prefix}.value"), n_state, n_state)?;
    let o = linear(wm, &format!("{prefix}.out"), n_state, n_state)?;
    Ok(MultiHeadAttention::new(q, k, v, o, n_head))
}

fn block<B: Backend>(
    wm: &mut WeightMap<B>,
    base: &str,
    n_state: usize,
    n_mlp: usize,
    n_head: usize,
    cross: bool,
) -> Result<ResidualAttentionBlock<B>> {
    let attn = mha(wm, &format!("{base}.attn"), n_state, n_head)?;
    let attn_ln = ln(wm, &format!("{base}.attn_ln"), n_state)?;
    let mlp = linear(wm, &format!("{base}.mlp"), n_mlp, n_state)?;
    let mlp_ln = ln(wm, &format!("{base}.mlp_ln"), n_state)?;
    let mlp2 = linear(wm, &format!("{base}.mlp2"), n_state, n_mlp)?;
    let lnb = ln_opt(wm, &format!("{base}.ln"), n_state)?;
    let mut xa_attn = None;
    let mut xa_attn_ln = None;
    let xa_ln = if cross {
        xa_attn = Some(mha(wm, &format!("{base}.xa_attn"), n_state, n_head)?);
        xa_attn_ln = Some(ln(wm, &format!("{base}.xa_attn_ln"), n_state)?);
        // present in the OpenAI .pt state dict, absent in HF safetensors, and
        // unused in forward: consume it when present for a strict drain.
        match wm.take_opt_2d(&format!("{base}.xa_ln.weight"), [n_state, n_state])? {
            Some(w) => Some(Linear {
                weight: Param::from_tensor(w),
                bias: None,
            }),
            None => None,
        }
    } else {
        None
    };
    Ok(ResidualAttentionBlock {
        attn,
        attn_ln,
        mlp,
        mlp_ln,
        mlp2,
        ln: lnb,
        xa_attn,
        xa_attn_ln,
        xa_ln,
    })
}

fn encoder<B: Backend>(wm: &mut WeightMap<B>, d: &ModelDimensions) -> Result<Encoder<B>> {
    let conv1 = Conv1d {
        weight: Param::from_tensor(
            wm.take_3d("encoder.conv1.weight", [d.n_audio_state, d.n_mels, 3])?,
        ),
        bias: Param::from_tensor(wm.take_1d("encoder.conv1.bias", d.n_audio_state)?),
    };
    let conv2 = Conv1d {
        weight: Param::from_tensor(wm.take_3d(
            "encoder.conv2.weight",
            [d.n_audio_state, d.n_audio_state, 3],
        )?),
        bias: Param::from_tensor(wm.take_1d("encoder.conv2.bias", d.n_audio_state)?),
    };
    let positional_embedding = Param::from_tensor(wm.take_2d(
        "encoder.positional_embedding",
        [d.n_audio_ctx, d.n_audio_state],
    )?);
    let blocks = (0..d.n_audio_layer)
        .map(|i| {
            block(
                wm,
                &format!("encoder.blocks.{i}"),
                d.n_audio_state,
                d.n_audio_state * 4,
                d.n_head,
                false,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ln_post = ln(wm, "encoder.ln_post", d.n_audio_state)?;
    Ok(Encoder {
        conv1,
        conv2,
        positional_embedding,
        blocks,
        ln_post,
    })
}

fn decoder<B: Backend>(
    wm: &mut WeightMap<B>,
    d: &ModelDimensions,
    n_text_ctx: usize,
) -> Result<Decoder<B>> {
    let token_embedding = Param::from_tensor(wm.take_2d(
        "decoder.token_embedding.weight",
        [d.n_vocab, d.n_text_state],
    )?);
    let positional_embedding = Param::from_tensor(
        wm.take_2d("decoder.positional_embedding", [n_text_ctx, d.n_text_state])?,
    );
    let blocks = (0..d.n_text_layer)
        .map(|i| {
            block(
                wm,
                &format!("decoder.blocks.{i}"),
                d.n_text_state,
                d.n_text_state * 4,
                d.n_head,
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ln = ln(wm, "decoder.ln", d.n_text_state)?;
    Ok(Decoder {
        token_embedding,
        positional_embedding,
        blocks,
        ln,
    })
}

/// Full encoder+decoder assembled from a `WeightMap`. Every key in the map must
/// be consumed (either here or by the caller) before `WeightMap::finish`.
#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    pub dims: ModelDimensions,
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn from_weights(
        dims: ModelDimensions,
        wm: &mut WeightMap<B>,
        n_text_ctx: usize,
    ) -> Result<Self> {
        let encoder = encoder(wm, &dims)?;
        let decoder = decoder(wm, &dims, n_text_ctx)?;
        Ok(Self {
            dims,
            encoder,
            decoder,
        })
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 1, burn::tensor::Int>,
        xa: &Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        self.decoder.forward(tokens, xa.clone())
    }

    /// Loads a checkpoint from a local directory: `config.json` validated
    /// against `size`, then `model.safetensors` drained into the model.
    pub fn load(size: ModelSize, weights_dir: &Path, device: B::Device) -> Result<Self> {
        use crate::weights::loader::WeightMap;
        use crate::weights::safetensors::read_safetensors;

        let st_path = weights_dir.join("model.safetensors");
        if !st_path.exists() {
            return Err(Error::MissingWeights { path: st_path });
        }
        let dims = read_config(&weights_dir.join("config.json"), size)?;
        let bytes = std::fs::read(&st_path).map_err(|source| Error::Io {
            path: st_path,
            source,
        })?;
        log::info!(
            "weights: reading {} ({:.0} MB)",
            weights_dir.join("model.safetensors").display(),
            bytes.len() as f64 / (1024.0 * 1024.0)
        );
        let st = read_safetensors(&bytes)?;
        log::info!(
            "model: {} ({} blocks, n_mels {}, n_ctx {})",
            size.repo_id(),
            dims.n_audio_state / 64,
            dims.n_mels,
            dims.n_text_ctx
        );
        log::info!("weights: {} tensors loaded", st.tensors.len());
        let mut wm = WeightMap::new(st, bytes, device);
        let n_text_ctx = dims.n_text_ctx;
        let model = Self::from_weights(dims, &mut wm, n_text_ctx)?;
        wm.finish()?;
        Ok(model)
    }

    /// Loads a checkpoint from the model cache
    /// (`WHISPER_BURN_CACHE` override or `~/.cache/whisper-burn`), downloading
    /// it first when the `weights` feature is enabled and files are missing.
    #[cfg(feature = "weights")]
    pub fn from_pretrained(size: ModelSize, device: B::Device) -> Result<Self> {
        Self::from_pretrained_with_progress(size, device, None)
    }

    /// Like [`Whisper::from_pretrained`] but reports checkpoint download
    /// progress through `on_progress` (`downloaded` bytes, `total` when known).
    #[cfg(feature = "weights")]
    pub fn from_pretrained_with_progress(
        size: ModelSize,
        device: B::Device,
        on_progress: Option<crate::download::DownloadCallback>,
    ) -> Result<Self> {
        let dir = crate::download::cache_dir(size)?;
        if !dir.join("model.safetensors").exists() {
            log::info!("weights: {} not cached — fetching", size.repo_id());
            crate::download::download_checkpoint_with_progress(size, &dir, true, true, on_progress)?;
        } else {
            log::info!("weights: {} cached at {}", size.repo_id(), dir.display());
        }
        Self::load(size, &dir, device)
    }

    /// Available without the `weights` feature; reports how to enable downloads.
    #[cfg(not(feature = "weights"))]
    pub fn from_pretrained(_size: ModelSize, _device: B::Device) -> Result<Self> {
        Err(Error::MissingWeights {
            path: Path::new("(weights feature disabled; enable the `weights` feature to download)")
                .into(),
        })
    }
}
