use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::Error;
use whisper_burn::config::ModelDimensions;
use whisper_burn::model::whisper::Whisper;
use whisper_burn::weights::loader::WeightMap;
use whisper_burn::weights::safetensors::read_safetensors;

type B = NdArray<f32>;

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
        1.0,
        skip,
    );
    push(json, payload, &format!("{prefix}.bias"), &[out], 1.0, skip);
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
        1.0,
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
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.attn_ln.bias"),
        &[s],
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp.weight"),
        &[4 * s, s],
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp.bias"),
        &[4 * s],
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp_ln.weight"),
        &[s],
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp_ln.bias"),
        &[s],
        1.0,
        skip,
    );
    push(
        json,
        payload,
        &format!("{base}.mlp2.weight"),
        &[s, 4 * s],
        1.0,
        skip,
    );
    push(json, payload, &format!("{base}.mlp2.bias"), &[s], 1.0, skip);
    push(json, payload, &format!("{base}.ln.weight"), &[s], 1.0, skip);
    push(json, payload, &format!("{base}.ln.bias"), &[s], 1.0, skip);
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
            1.0,
            skip,
        );
        push(
            json,
            payload,
            &format!("{base}.xa_attn_ln.bias"),
            &[s],
            1.0,
            skip,
        );
        // bias-less, present in the state dict, unused in forward: must still drain
        push(
            json,
            payload,
            &format!("{base}.xa_ln.weight"),
            &[s, s],
            1.0,
            skip,
        );
    }
}

/// Full weight set for `d` with all values 1.0 (so forwards stay finite).
fn synthetic_bytes(
    d: &ModelDimensions,
    skip: Option<&str>,
    vocab_override: Option<usize>,
) -> Vec<u8> {
    let mut json = String::new();
    let mut payload = Vec::new();
    let s = d.n_audio_state;
    let st = d.n_text_state;
    let vocab = vocab_override.unwrap_or(d.n_vocab);

    push(
        &mut json,
        &mut payload,
        "encoder.conv1.weight",
        &[s, d.n_mels, 3],
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv1.bias",
        &[s],
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv2.weight",
        &[s, s, 3],
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.conv2.bias",
        &[s],
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.positional_embedding",
        &[d.n_audio_ctx, s],
        1.0,
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
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "encoder.ln_post.bias",
        &[s],
        1.0,
        skip,
    );

    push(
        &mut json,
        &mut payload,
        "decoder.token_embedding.weight",
        &[vocab, st],
        1.0,
        skip,
    );
    push(
        &mut json,
        &mut payload,
        "decoder.positional_embedding",
        &[d.n_text_ctx, st],
        1.0,
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
        1.0,
        skip,
    );
    push(&mut json, &mut payload, "decoder.ln.bias", &[st], 1.0, skip);

    build_file(&json, &payload)
}

fn tiny_dims() -> ModelDimensions {
    ModelDimensions {
        n_mels: 16,
        n_audio_layer: 2,
        n_text_layer: 2,
        n_audio_state: 32,
        n_text_state: 32,
        n_head: 4,
        n_vocab: 64,
        n_audio_ctx: 1500,
        n_text_ctx: 448,
    }
}

#[test]
fn synthetic_checkpoint_loads_and_drains() {
    let d = tiny_dims();
    let bytes = synthetic_bytes(&d, None, None);
    let st = read_safetensors(&bytes).unwrap();
    let mut wm = WeightMap::<B>::new(st, bytes, NdArrayDevice::default());
    let w = Whisper::from_weights(d.clone(), &mut wm, d.n_text_ctx).unwrap();

    assert_eq!(
        w.encoder.conv1.weight.val().shape().dims(),
        [d.n_audio_state, d.n_mels, 3]
    );
    assert_eq!(
        w.decoder.token_embedding.val().shape().dims(),
        [d.n_vocab, d.n_text_state]
    );
    assert_eq!(w.encoder.blocks.len(), d.n_audio_layer);
    assert_eq!(w.decoder.blocks.len(), d.n_text_layer);
    // every key was consumed
    wm.finish().unwrap();

    let device = NdArrayDevice::default();
    let mel: Tensor<B, 3> = Tensor::zeros([1, d.n_mels, 3000], &device);
    let fe = w.forward_encoder(mel);
    assert_eq!(fe.shape().dims(), [1, d.n_audio_ctx, d.n_audio_state]);
    assert!(
        fe.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );

    let tokens = Tensor::<B, 1>::from_data(TensorData::from([0.0f32, 1.0]), &device).int();
    let xa: Tensor<B, 3> = Tensor::zeros([1, 100, d.n_text_state], &device);
    let lg = w.forward_decoder(tokens, &xa);
    assert_eq!(lg.shape().dims(), [2, d.n_vocab]);
    assert!(
        lg.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn missing_weight_errors_on_from_weights() {
    let d = tiny_dims();
    let bytes = synthetic_bytes(&d, Some("decoder.blocks.1.xa_attn.out.bias"), None);
    let st = read_safetensors(&bytes).unwrap();
    let mut wm = WeightMap::<B>::new(st, bytes, NdArrayDevice::default());
    match Whisper::from_weights(d.clone(), &mut wm, d.n_text_ctx) {
        Err(Error::MissingWeight(name)) => {
            assert!(name.contains("xa_attn.out.bias"), "unexpected key {name}");
        }
        other => panic!("expected MissingWeight, got {other:?}"),
    }
}

#[test]
fn wrong_shape_weight_errors() {
    let d = tiny_dims();
    let bytes = synthetic_bytes(&d, None, Some(d.n_vocab + 1));
    let st = read_safetensors(&bytes).unwrap();
    let mut wm = WeightMap::<B>::new(st, bytes, NdArrayDevice::default());
    match Whisper::from_weights(d.clone(), &mut wm, d.n_text_ctx) {
        Err(Error::ShapeMismatch { name, .. }) => {
            assert!(name.contains("token_embedding"), "unexpected key {name}");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}
