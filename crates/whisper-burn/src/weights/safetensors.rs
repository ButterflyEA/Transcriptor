use crate::{Error, Result};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        })
    }
}

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct SafeTensors {
    pub metadata: HashMap<String, String>,
    pub tensors: Vec<TensorMeta>,
}

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    data_offsets: Option<[usize; 2]>,
}

impl SafeTensors {
    pub fn by_name(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn all_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }
}

/// Normalize a checkpoint tensor name to the canonical whisper state-dict
/// naming used by the loaders. Handles the two conventions found in the wild:
/// * a leading `model.` prefix (HF safetensors exports), and
/// * fairseq-style names (`layers.{i}`, `self_attn`, `fc1/fc2`,
///   `final_layer_norm`, `embed_positions`, `embed_tokens`, `encoder_attn`,
///   `layer_norm`) instead of OpenAI-style (`blocks.{i}`, `attn`, `mlp/mlp2`,
///   `ln`, `positional_embedding`, `token_embedding`, `xa_attn`, `ln_post`).
///
/// The function is idempotent for already-canonical names.
pub fn canonical_key(key: &str) -> String {
    let parts: Vec<&str> = key
        .strip_prefix("model.")
        .unwrap_or(key)
        .split('.')
        .collect();
    let out = parts
        .iter()
        .map(|p| match *p {
            "layers" => "blocks",
            "self_attn" => "attn",
            "self_attn_layer_norm" => "attn_ln",
            "encoder_attn" => "xa_attn",
            "encoder_attn_layer_norm" => "xa_attn_ln",
            "fc1" => "mlp",
            "fc2" => "mlp2",
            "final_layer_norm" => "mlp_ln",
            "q_proj" => "query",
            "k_proj" => "key",
            "v_proj" => "value",
            "out_proj" => "out",
            p => p,
        })
        .collect::<Vec<_>>()
        .join(".");
    match out.as_str() {
        "encoder.embed_positions.weight" => "encoder.positional_embedding".into(),
        "decoder.embed_positions.weight" => "decoder.positional_embedding".into(),
        "decoder.embed_tokens.weight" => "decoder.token_embedding.weight".into(),
        "encoder.layer_norm.weight" => "encoder.ln_post.weight".into(),
        "encoder.layer_norm.bias" => "encoder.ln_post.bias".into(),
        "decoder.layer_norm.weight" => "decoder.ln.weight".into(),
        "decoder.layer_norm.bias" => "decoder.ln.bias".into(),
        _ => out,
    }
}

pub fn read_safetensors(bytes: &[u8]) -> Result<SafeTensors> {
    if bytes.len() < 8 {
        return Err(Error::Download("safetensors: too short".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let json_start: usize = 8;
    let json_end = json_start
        .checked_add(n)
        .ok_or_else(|| Error::Download("safetensors: header overflow".into()))?;
    if json_end > bytes.len() {
        return Err(Error::Download("safetensors: truncated header".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[json_start..json_end])
        .map_err(|e| Error::Download(format!("safetensors header: {e}")))?;
    let obj = header
        .as_object()
        .ok_or_else(|| Error::Download("safetensors: header not object".into()))?;

    let mut metadata = HashMap::new();
    if let Some(m) = obj.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                metadata.insert(k.clone(), s.to_string());
            }
        }
    }

    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let e: HeaderEntry = serde_json::from_value(v.clone())
            .map_err(|e| Error::Download(format!("safetensors tensor {name}: {e}")))?;
        let dtype = match e.dtype.as_deref() {
            Some("F32") => DType::F32,
            Some("F16") => DType::F16,
            Some("BF16") => DType::BF16,
            other => {
                return Err(Error::UnsupportedFormat(format!(
                    "safetensors dtype {other:?} for {name}"
                )));
            }
        };
        let shape = e
            .shape
            .ok_or_else(|| Error::Download(format!("safetensors {name}: no shape")))?;
        let off = e
            .data_offsets
            .ok_or_else(|| Error::Download(format!("safetensors {name}: no offsets")))?;
        tensors.push(TensorMeta {
            name: canonical_key(name),
            dtype,
            shape,
            data_offsets: (off[0], off[1]),
        });
    }

    Ok(SafeTensors { metadata, tensors })
}

/// Frozen-precision to `f32` conversion. Tensors are always exposed as `f32`
/// (matching upstream whisper, which loads every checkpoint and casts
/// `.float()`); F16/BF16 checkpoints are converted on read.
///
/// `F32` borrows the input bytes (zero-copy); `F16`/`BF16` return an owned
/// vector of converted values.
pub fn slice_f32<'a>(meta: &TensorMeta, bytes: &'a [u8]) -> Result<Cow<'a, [f32]>> {
    if bytes.len() < 8 {
        return Err(Error::Download("safetensors: too short".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    // The header is padded to an 8-byte boundary; data offsets are relative to it.
    let data_start = (8 + n).div_ceil(8) * 8;
    let (a, b) = meta.data_offsets;
    let start = data_start
        .checked_add(a)
        .ok_or_else(|| Error::Download("offset overflow".into()))?;
    let end = data_start
        .checked_add(b)
        .ok_or_else(|| Error::Download("offset overflow".into()))?;
    if end > bytes.len() {
        return Err(Error::Download("tensor out of bounds".into()));
    }
    let data = &bytes[start..end];

    match meta.dtype {
        DType::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(Error::UnsupportedFormat(format!(
                    "tensor {} byte length {} not multiple of 4",
                    meta.name,
                    data.len()
                )));
            }
            if !(bytes.as_ptr() as usize + start).is_multiple_of(std::mem::align_of::<f32>()) {
                return Err(Error::UnsupportedFormat(format!(
                    "tensor {} data at byte {} is not 4-aligned",
                    meta.name, start
                )));
            }
            let ptr = data.as_ptr() as *const f32;
            Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(ptr, data.len() / 4)
            }))
        }
        DType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(Error::UnsupportedFormat(format!(
                    "tensor {} byte length {} not multiple of 2",
                    meta.name,
                    data.len()
                )));
            }
            let vals = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| f16_to_f32(u16::from_le_bytes(*p)))
                .collect();
            Ok(Cow::Owned(vals))
        }
        DType::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(Error::UnsupportedFormat(format!(
                    "tensor {} byte length {} not multiple of 2",
                    meta.name,
                    data.len()
                )));
            }
            let vals = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| bf16_to_f32(u16::from_le_bytes(*p)))
                .collect();
            Ok(Cow::Owned(vals))
        }
    }
}

/// IEEE 754 half-precision to single precision. Preserves ±0, ±Inf, NaN,
/// and converts denormals to the equivalent normal `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h >> 10) & 0x1F;
    let man = (h & 0x03FF) as u32;
    let bits = match exp {
        0x00 if man == 0 => sign, // ±0.0
        0x00 => {
            // Denormal: value = man * 2^-24. Shift to a normal f32.
            let mut m = man;
            let mut e = 113u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (m & 0x3FF) << 13 | (e << 23) | sign
        }
        0x1F => 0x7F80_0000 | (man << 13) | sign, // ±Inf / NaN
        e => ((e as u32 + 112) << 23) | (man << 13) | sign,
    };
    f32::from_bits(bits)
}

/// bfloat16 to single precision: the bf16 bits are the high 16 of the f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Combines the shard files of a sharded safetensors checkpoint (HF
/// `model-00001-of-0000N.safetensors` layout) into a single-file checkpoint.
/// Each shard's data section is concatenated and every tensor's
/// `data_offsets` are rebased onto the merged buffer; tensor names and dtypes
/// pass through unchanged (`read_safetensors` canonicalizes them later).
///
/// The whole contents are held in memory, so this roughly doubles peak RAM
/// for the duration of a merge (about 6 GB for a 3 GB large-v3 model).
pub fn merge_safetensors(shards: &[Vec<u8>]) -> Result<Vec<u8>> {
    if shards.is_empty() {
        return Err(Error::Download("safetensors: no shards to merge".into()));
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for shard in shards {
        if shard.len() < 8 {
            return Err(Error::Download("safetensors: shard too short".into()));
        }
        let n = u64::from_le_bytes(shard[..8].try_into().expect("8 bytes")) as usize;
        let json_end = 8usize
            .checked_add(n)
            .ok_or_else(|| Error::Download("safetensors: header overflow".into()))?;
        if json_end > shard.len() {
            return Err(Error::Download("safetensors: truncated shard header".into()));
        }
        let value: serde_json::Value = serde_json::from_slice(&shard[8..json_end])
            .map_err(|e| Error::Download(format!("safetensors shard header: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| Error::Download("safetensors: shard header not object".into()))?;
        let data_start = (8 + n).div_ceil(8) * 8;
        let mut data_len = 0usize;
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    header.insert(name.clone(), serde_json::Value::Object(m.clone()));
                }
                continue;
            }
            let e: HeaderEntry = serde_json::from_value(v.clone())
                .map_err(|e| Error::Download(format!("safetensors shard tensor {name}: {e}")))?;
            let dtype = e
                .dtype
                .ok_or_else(|| Error::Download(format!("safetensors {name}: no dtype")))?;
            let shape = e
                .shape
                .ok_or_else(|| Error::Download(format!("safetensors {name}: no shape")))?;
            let [a, b] = e.data_offsets.ok_or_else(|| {
                Error::Download(format!("safetensors {name}: no data offsets"))
            })?;
            if a >= b || data_start + b > shard.len() {
                return Err(Error::Download(format!(
                    "safetensors {name}: offsets out of bounds"
                )));
            }
            data_len = data_len.max(b);
            let base = data.len();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [base + a, base + b],
                }),
            );
        }
        data.extend_from_slice(&shard[data_start..data_start + data_len]);
    }

    let json = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| Error::Download(format!("safetensors: serialize merged header: {e}")))?;
    // Header length padded so the data section starts 8-byte aligned.
    let n = (8 + json.len()).div_ceil(8) * 8 - 8;
    let mut out = Vec::with_capacity(8 + n + data.len());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.resize(8 + n, b' ');
    out.extend_from_slice(&data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a single-file safetensors binary from `(name, values)` pairs.
    fn st_bytes(tensors: &[(&str, &[f32])]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut map = serde_json::Map::new();
        let mut off = 0usize;
        for (name, vals) in tensors {
            let n = vals.len() * 4;
            map.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [vals.len()],
                    "data_offsets": [off, off + n],
                }),
            );
            for v in *vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            off += n;
        }
        let json = serde_json::to_vec(&map).expect("serialize header");
        // Header length padded so the data section starts 8-byte aligned.
        let n = (8 + json.len()).div_ceil(8) * 8 - 8;
        let mut out = Vec::new();
        out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(&json);
        out.resize(8 + n, b' ');
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn merge_combines_tensor_data_across_shards() {
        let shard_a = st_bytes(&[("a.weight", &[1.0, 2.0])]);
        let shard_b = st_bytes(&[("b.weight", &[3.0, 4.0, 5.0])]);
        let merged = merge_safetensors(&[shard_a, shard_b]).expect("merge");
        let st = read_safetensors(&merged).expect("parse merged");
        assert_eq!(st.tensors.len(), 2);
        let a = st.by_name("a.weight").expect("a");
        assert_eq!(a.shape, vec![2]);
        assert_eq!(&*slice_f32(a, &merged).expect("a data"), &[1.0, 2.0]);
        let b = st.by_name("b.weight").expect("b");
        assert_eq!(b.shape, vec![3]);
        assert_eq!(&*slice_f32(b, &merged).expect("b data"), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn merge_handles_multiple_tensors_per_shard() {
        let shard = st_bytes(&[("x.weight", &[10.0]), ("y.weight", &[20.0])]);
        let merged = merge_safetensors(&[shard]).expect("merge single");
        let st = read_safetensors(&merged).expect("parse");
        assert_eq!(&*slice_f32(st.by_name("x.weight").expect("x"), &merged).expect("x"), &[10.0]);
        assert_eq!(&*slice_f32(st.by_name("y.weight").expect("y"), &merged).expect("y"), &[20.0]);
    }
}
