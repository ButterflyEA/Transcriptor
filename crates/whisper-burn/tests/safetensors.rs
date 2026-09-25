use whisper_burn::weights::safetensors::{self, DType};

fn build(json: &str, payload: &[u8]) -> Vec<u8> {
    // safetensors pads the JSON header to an 8-byte boundary.
    let n: u64 = (json.len().div_ceil(8) * 8) as u64;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.resize(8 + n as usize, b' ');
    out.extend_from_slice(payload);
    out
}

#[test]
fn parses_single_f32_tensor() {
    let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
    let payload = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("w").unwrap();
    assert_eq!(t.dtype.to_string(), "F32");
    assert_eq!(t.shape, vec![2, 3]);
    let f = safetensors::slice_f32(t, &file).unwrap();
    assert_eq!(&*f, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn converts_f16_tensor_to_f32() {
    // IEEE 754 half values: 1.0, 0.5, -2.0, 0.0, 65504.0 (f16 max),
    // and a subnormal (2^-24).
    let half: [u16; 6] = [0x3C00, 0x3800, 0xC000, 0x0000, 0x7BFF, 0x0001];
    let payload = half
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let json = r#"{"w": {"dtype": "F16", "shape": [6], "data_offsets": [0, 12]}}"#;
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("w").unwrap();
    assert!(matches!(t.dtype, DType::F16));
    let f = safetensors::slice_f32(t, &file).unwrap();
    assert_eq!(&*f, &[1.0, 0.5, -2.0, 0.0, 65504.0, 2f32.powi(-24)]);
}

#[test]
fn converts_bf16_tensor_to_f32() {
    // bfloat16: high 16 bits of the f32 bit pattern.
    let bf16: [u16; 4] = [0x3F80, 0xBF80, 0x0000, 0x3F00]; // 1.0, -1.0, 0.0, 0.5
    let payload = bf16
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let json = r#"{"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}"#;
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("w").unwrap();
    assert!(matches!(t.dtype, DType::BF16));
    let f = safetensors::slice_f32(t, &file).unwrap();
    assert_eq!(&*f, &[1.0, -1.0, 0.0, 0.5]);
}

#[test]
fn strips_openai_model_prefix() {
    // HF-exported openai/whisper safetensors prefix every key with "model.".
    let json = r#"{"model.encoder.conv1.weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"#;
    let payload = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let file = build(json, &payload);
    let st = safetensors::read_safetensors(&file).unwrap();
    let t = st.by_name("encoder.conv1.weight").unwrap();
    assert_eq!(t.shape, vec![4]);
    let f = safetensors::slice_f32(t, &file).unwrap();
    assert_eq!(&*f, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn canonicalizes_fairseq_whisper_key_names() {
    // openai/whisper-tiny on HF ships fairseq-style names (embed_positions,
    // self_attn, fc1/fc2, final_layer_norm, layers.*) which differ from the
    // canonical whisper state dict used by the loaders.
    let entries = [
        r#""model.encoder.embed_positions.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layer_norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.self_attn.q_proj.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.self_attn.out_proj.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.self_attn_layer_norm.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.fc1.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.fc2.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.layers.0.final_layer_norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.decoder.embed_tokens.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.decoder.embed_positions.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.decoder.layer_norm.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.decoder.layers.3.encoder_attn.k_proj.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.decoder.layers.3.encoder_attn_layer_norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
        r#""model.encoder.conv1.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}"#,
    ];
    let json = format!("{{{}}}", entries.join(","));
    let file = build(&json, &[0u8; 4]);
    let st = safetensors::read_safetensors(&file).unwrap();
    let want = [
        "encoder.positional_embedding",
        "encoder.ln_post.weight",
        "encoder.blocks.0.attn.query.weight",
        "encoder.blocks.0.attn.out.bias",
        "encoder.blocks.0.attn_ln.bias",
        "encoder.blocks.0.mlp.weight",
        "encoder.blocks.0.mlp2.bias",
        "encoder.blocks.0.mlp_ln.weight",
        "decoder.token_embedding.weight",
        "decoder.positional_embedding",
        "decoder.ln.bias",
        "decoder.blocks.3.xa_attn.key.weight",
        "decoder.blocks.3.xa_attn_ln.weight",
        "encoder.conv1.weight",
    ];
    let got: Vec<&str> = st.all_tensor_names().collect();
    for w in want {
        assert!(got.contains(&w), "missing canonical key {w}");
    }
    assert_eq!(got.len(), want.len(), "unexpected extras: {got:?}");
}

#[test]
fn rejects_malformed() {
    assert!(safetensors::read_safetensors(&[0u8; 8]).is_err());
    assert!(safetensors::read_safetensors(b"nope").is_err());
}
