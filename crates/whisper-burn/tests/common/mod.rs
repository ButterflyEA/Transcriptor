use std::path::Path;
use whisper_burn::config::{ModelDimensions, ModelSize};

/// Real whisper-tiny dimensions, matching `tests/fixtures/tiny_config.json`.
pub fn tiny_dims() -> ModelDimensions {
    whisper_burn::config::read_config("tests/fixtures/tiny_config.json".as_ref(), ModelSize::Tiny)
        .unwrap()
}

/// Build a safetensors buffer (json header padded to 8 bytes, then payload).
fn build_file(json: &str, payload: &[u8]) -> Vec<u8> {
    let json = format!("{{{json}}}");
    let n: u64 = (json.len().div_ceil(8) * 8) as u64;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.resize(8 + n as usize, b' ');
    out.extend_from_slice(payload);
    out
}

/// Append one F32 tensor of `fill` values; data_offsets relative to data start.
fn push(
    json: &mut String,
    payload: &mut Vec<u8>,
    name: &str,
    shape: &[usize],
    fill: f32,
    skip: Option<&str>,
) {
    if skip == Some(name) {
        return;
    }
    let n: usize = shape.iter().product();
    let start = payload.len();
    for _ in 0..n {
        payload.extend_from_slice(&fill.to_le_bytes());
    }
    let end = payload.len();
    if !json.is_empty() {
        json.push(',');
    }
    json.push_str(&format!(
        r#""{name}":{{"dtype":"F32","shape":{shape:?},"data_offsets":[{start},{end}]}}"#
    ));
}

fn tmp_linear(
    json: &mut String,
    payload: &mut Vec<u8>,
    prefix: &str,
    out: usize,
    nin: usize,
    skip: Option<&str>,
) {
    push(
        json,
        payload,
        &format!("{prefix}.weight"),
        &[out, nin],
        0.0,
        skip,
    );
    push(json, payload, &format!("{prefix}.bias"), &[out], 0.0, skip);
}

fn tmp_linear_nobias(
    json: &mut String,
    payload: &mut Vec<u8>,
    prefix: &str,
    out: usize,
    nin: usize,
    skip: Option<&str>,
) {
    push(
        json,
        payload,
        &format!("{prefix}.weight"),
        &[out, nin],
        0.0,
        skip,
    );
}

fn tmp_block(
    json: &mut String,
    payload: &mut Vec<u8>,
    base: &str,
    s: usize,
    cross: bool,
    skip: Option<&str>,
) {
    // reference whisper: key projection is Linear(..., bias=False)
    tmp_linear(json, payload, &format!("{base}.attn.query"), s, s, skip);
    tmp_linear_nobias(json, payload, &format!("{base}.attn.key"), s, s, skip);
    tmp_linear(json, payload, &format!("{base}.attn.value"), s, s, skip);
    tmp_linear(json, payload, &format!("{base}.attn.out"), s, s, skip);
    push(
        json,
        payload,
        &format!("{base}.attn_ln.weight"),
        &[s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.attn_ln.bias"),
        &[s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp.weight"),
        &[4 * s, s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp.bias"),
        &[4 * s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp_ln.weight"),
        &[s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp_ln.bias"),
        &[s],
        0.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp2.weight"),
        &[s, 4 * s],
        0.0,
        skip,
    );
    push(json, payload, &format!("{base}.mlp2.bias"), &[s], 0.0, skip);
    push(json, payload, &format!("{base}.ln.weight"), &[s], 0.0, skip);
    push(json, payload, &format!("{base}.ln.bias"), &[s], 0.0, skip);
    if cross {
        tmp_linear(json, payload, &format!("{base}.xa_attn.query"), s, s, skip);
        tmp_linear_nobias(json, payload, &format!("{base}.xa_attn.key"), s, s, skip);
        tmp_linear(json, payload, &format!("{base}.xa_attn.value"), s, s, skip);
        tmp_linear(json, payload, &format!("{base}.xa_attn.out"), s, s, skip);
        push(
            json,
            payload,
            &format!("{base}.xa_attn_ln.weight"),
            &[s],
            0.0,
            skip,
        );
        push(
            json,
            payload,
            &format!("{base}.xa_attn_ln.bias"),
            &[s],
            0.0,
            skip,
        );
        // NOTE: real HF tiny checkpoints have no separately-named `xa_ln` tensor
        // (the OpenAI .pt does); the loader consumes it optionally.
    }
}

/// Full tiny weight set with all values 0.0 (load only checks shape/drain).
fn synthetic_bytes(d: &ModelDimensions, skip: Option<&str>) -> Vec<u8> {
    let mut json = String::new();
    let mut payload = Vec::new();
    let s = d.n_audio_state;
    let st = d.n_text_state;

    push(
        &mut json,
        &mut payload,
        "encoder.conv1.weight",
        &[s, d.n_mels, 3],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv1.bias",
        &[s],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv2.weight",
        &[s, s, 3],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv2.bias",
        &[s],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.positional_embedding",
        &[d.n_audio_ctx, s],
        0.0,
        skip,
    );
    for i in 0..d.n_audio_layer {
        tmp_block(
            &mut json,
            &mut payload,
            &format!("encoder.blocks.{i}"),
            s,
            false,
            skip,
        );
    }
    push(
        &mut json,
        &mut payload,
        "encoder.ln_post.weight",
        &[s],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.ln_post.bias",
        &[s],
        0.0,
        skip,
    );

    push(
        &mut json,
        &mut payload,
        "decoder.token_embedding.weight",
        &[d.n_vocab, st],
        0.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "decoder.positional_embedding",
        &[d.n_text_ctx, st],
        0.0,
        skip,
    );
    for i in 0..d.n_text_layer {
        tmp_block(
            &mut json,
            &mut payload,
            &format!("decoder.blocks.{i}"),
            st,
            true,
            skip,
        );
    }
    push(
        &mut json,
        &mut payload,
        "decoder.ln.weight",
        &[st],
        0.0,
        skip,
    );
    push(&mut json, &mut payload, "decoder.ln.bias", &[st], 0.0, skip);

    build_file(&json, &payload)
}

const CONFIG_JSON: &str = include_str!("../fixtures/tiny_config.json");

fn write_checkpoint(dir: &Path, skip: Option<&str>) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();
    let bytes = synthetic_bytes(&tiny_dims(), skip);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

/// Writes a fake whisper-tiny checkpoint (`config.json` + `model.safetensors`,
/// all-zero values) into `dir`. Only `ModelSize::Tiny` is supported.
pub fn write_fake_checkpoint(size: ModelSize, dir: &Path) {
    assert!(size == ModelSize::Tiny, "only Tiny supported");
    write_checkpoint(dir, None);
}

/// Like `write_fake_checkpoint` but omits the given tensor key.
pub fn write_fake_checkpoint_without(size: ModelSize, dir: &Path, skip: &str) {
    assert!(size == ModelSize::Tiny, "only Tiny supported");
    write_checkpoint(dir, Some(skip));
}
