use crate::{Error, Result};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub fn decode_to_mono_f32(path: impl AsRef<std::path::Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let fmt_opts = FormatOptions::default();
    let meta_opts = MetadataOptions::default();
    let mut reader = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .map_err(|e| Error::AudioDecode(format!("probe {}: {e}", path.display())))?;

    let track = reader
        .default_track(TrackType::Audio)
        .ok_or_else(|| Error::AudioDecode("no default audio track".into()))?;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(CodecParameters::audio)
        .ok_or_else(|| Error::AudioDecode("track is not audio".into()))?;
    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode("unknown sample rate".into()))?;
    let channels = audio_params
        .channels
        .clone()
        .ok_or_else(|| Error::AudioDecode("unknown channel layout".into()))?;
    let n_ch = channels.count();

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| Error::AudioDecode(format!("codec: {e}")))?;

    let mut accum: Vec<Vec<f32>> = vec![Vec::new(); n_ch];
    while let Some(packet) = reader
        .next_packet()
        .map_err(|e| Error::AudioDecode(format!("packet: {e}")))?
    {
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(Error::AudioDecode(format!("decode: {e}"))),
        };
        let mut inter = Vec::with_capacity(buf.frames() * buf.spec().channels().count());
        buf.copy_to_vec_interleaved::<f32>(&mut inter);
        let n_ch = buf.spec().channels().count();
        for (i, &s) in inter.iter().enumerate() {
            accum[i % n_ch].push(s);
        }
    }

    let mut out = Vec::with_capacity(accum[0].len());
    for i in 0..accum[0].len() {
        let sum: f32 = accum.iter().map(|c| c[i]).sum();
        out.push(sum / n_ch as f32);
    }
    Ok((out, sample_rate))
}
