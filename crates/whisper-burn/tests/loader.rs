use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use whisper_burn::weights::loader::WeightMap;
use whisper_burn::weights::safetensors::read_safetensors;

type B = NdArray<f32>;

fn build_file(json: &str, payload: &[u8]) -> Vec<u8> {
    // safetensors pads the JSON header to an 8-byte boundary.
    let n: u64 = (json.len().div_ceil(8) * 8) as u64;
    let mut out = n.to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.resize(8 + n as usize, b' ');
    out.extend_from_slice(payload);
    out
}

fn flat_f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn drains_and_validates_weights() {
    let json = r#"{"enc.w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"enc.b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
    let mut payload = flat_f32s(&[1.0, 2.0, 3.0, 4.0]);
    payload.extend(flat_f32s(&[0.5, -0.5]));
    let bytes = build_file(json, &payload);
    let st = read_safetensors(&bytes).unwrap();
    let device = NdArrayDevice::default();
    let mut wm = WeightMap::<B>::new(st, bytes, device);
    let w: Tensor<B, 2> = wm.take_2d("enc.w", [2, 2]).unwrap();
    let b: Tensor<B, 1> = wm.take_1d("enc.b", 2).unwrap();
    assert_eq!(w.into_data(), TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]));
    assert_eq!(b.into_data(), TensorData::from([0.5f32, -0.5]));
    wm.finish().unwrap();
}

#[test]
fn rejects_leftover_and_wrong_shape() {
    let json = r#"{"enc.w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}} "#;
    let bytes = build_file(json, &flat_f32s(&[1.0, 2.0, 3.0, 4.0]));
    // leftover: consume nothing → finish must error
    let st = read_safetensors(&bytes).unwrap();
    let wm = WeightMap::<B>::new(st, bytes.clone(), NdArrayDevice::default());
    assert!(wm.finish().is_err()); // leftover enc.w
    // wrong shape on take
    let mut wm2 = WeightMap::<B>::new(
        read_safetensors(&bytes).unwrap(),
        bytes.clone(),
        NdArrayDevice::default(),
    );
    assert!(wm2.take_2d("enc.w", [3, 3]).is_err());
    assert!(wm2.take_2d("enc.w", [2, 2]).is_ok());
    // double-consumption rejected
    assert!(wm2.take_2d("enc.w", [2, 2]).is_err());
    wm2.finish().unwrap();
}
