use crate::tokenizer::tiktoken::CoreBpe;
use crate::{Error, Result};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    Translate,
}

// Special-token ids of the reference multilingual vocab. The 50257-entry base
// vocab is ranks 0..50256 (50256 = empty token); the ids above it depend on
// the checkpoint family (verified against openai's `tokenizer.json`):
// - pre-large-v3 (`n_vocab = 51865`): 50257 eot, 50258 sot, 50259..50357 the
//   99 languages, 50358 translate, 50359 transcribe, 50360 startoflm, 50361
//   startofprev, 50362 nospeech, 50363 notimestamps, 50364..51864 timestamps.
// - large-v3 / large-v3-turbo (`n_vocab = 51866`): `<|yue|>` becomes the 100th
//   language token (50358), shifting translate/transcribe and everything above
//   them up by one: 50359 translate, 50360 transcribe, 50361 startoflm, 50362
//   startofprev, 50363 nospeech, 50364 notimestamps, 50365..51865 timestamps.
// The constants below are the pre-large-v3 layout; `special_ids` selects the
// layout for a given `n_vocab`.
pub const EOT: u32 = 50257;
pub const SOT: u32 = 50258;
pub const TRANSLATE: u32 = 50358;
pub const TRANSCRIBE: u32 = 50359;
pub const STARTOFLM: u32 = 50360;
pub const STARTOPREV: u32 = 50361;
pub const NOSPEECH: u32 = 50362;
pub const NOTIMESTAMPS: u32 = 50363;
pub const LANGUAGE_BASE: u32 = 50259;
pub const TIMESTAMP_BEGIN: u32 = 50364;

/// The special-token positions for a checkpoint's vocab size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialIds {
    pub eot: u32,
    pub sot: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub startoflm: u32,
    pub startofprev: u32,
    pub nospeech: u32,
    pub notimestamps: u32,
    pub timestamp_begin: u32,
    /// Language tokens: 99 standard codes, 100 with `yue` for large-v3.
    pub n_languages: usize,
}

/// Select the special-token layout for a checkpoint: large-v3 / large-v3-turbo
/// (`n_vocab = 51866`) uses the shifted v3 layout, everything else the v2
/// `multilingual` layout.
pub fn special_ids(n_vocab: usize) -> SpecialIds {
    if n_vocab == 51866 {
        SpecialIds {
            eot: EOT,
            sot: SOT,
            translate: 50359,
            transcribe: 50360,
            startoflm: 50361,
            startofprev: 50362,
            nospeech: 50363,
            notimestamps: 50364,
            timestamp_begin: 50365,
            n_languages: 100,
        }
    } else {
        SpecialIds {
            eot: EOT,
            sot: SOT,
            translate: TRANSLATE,
            transcribe: TRANSCRIBE,
            startoflm: STARTOFLM,
            startofprev: STARTOPREV,
            nospeech: NOSPEECH,
            notimestamps: NOTIMESTAMPS,
            timestamp_begin: TIMESTAMP_BEGIN,
            n_languages: LANGUAGES.len(),
        }
    }
}

/// The language-token ids `<|{code}|>` for a given vocab size, in `LANGUAGES`
/// order plus `yue` from slot 100 (large-v3).
pub fn language_codes(n: usize) -> Vec<&'static str> {
    let mut codes: Vec<&'static str> = LANGUAGES.to_vec();
    if n > codes.len() {
        codes.push("yue");
    }
    codes.truncate(n);
    codes
}

/// The control tokens decoding must never sample: `{translate, transcribe,
/// sot, sot_prev, sot_lm, nospeech}` (reference `_get_suppress_tokens`).
pub fn control_tokens(n_vocab: usize) -> Vec<u32> {
    let s = special_ids(n_vocab);
    vec![
        s.translate,
        s.transcribe,
        s.sot,
        s.startofprev,
        s.startoflm,
        s.nospeech,
    ]
}

/// Whisper's 99 languages, sorted by code (matches reference `LANGUAGES`).
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

pub struct TextTokenizer {
    pub core: CoreBpe,
    pub n_vocab: u32,
    pub n_audio_ctx: usize,
    pub timestamp_begin: u32,
    pub special: SpecialIds,
    pub language_ids: HashMap<&'static str, u32>,
}

fn make_specials(n_vocab: u32) -> Vec<(String, u32)> {
    let s = special_ids(n_vocab as usize);
    let mut v = vec![
        ("<|endoftext|>".to_string(), s.eot),
        ("<|startoftranscript|>".to_string(), s.sot),
        ("<|translate|>".to_string(), s.translate),
        ("<|transcribe|>".to_string(), s.transcribe),
        ("<|startoflm|>".to_string(), s.startoflm),
        ("<|startofprev|>".to_string(), s.startofprev),
        ("<|nospeech|>".to_string(), s.nospeech),
        ("<|notimestamps|>".to_string(), s.notimestamps),
    ];
    for (i, code) in language_codes(s.n_languages).into_iter().enumerate() {
        v.push((format!("<|{code}|>"), LANGUAGE_BASE + i as u32));
    }
    v.push(("<|0.00|>".to_string(), s.timestamp_begin));
    v
}

impl TextTokenizer {
    pub const LANGUAGES: &[&str] = LANGUAGES;

    pub fn new(bytes: &[u8], n_vocab: u32, n_audio_ctx: usize) -> Result<Self> {
        let specials = make_specials(n_vocab);
        let core = CoreBpe::from_tiktoken_with_special(bytes, &specials)?;
        Self::new_from_core(core, n_vocab, n_audio_ctx)
    }

    /// The standard reference tokenizer: the `multilingual.tiktoken` byte-level
    /// BPE vocab shipped with openai/whisper (MIT), embedded so `transcribe`
    /// works from any load path. Multilingual checkpoints only; the
    /// English-only (`gpt2`) vocab is out of scope for now.
    pub fn standard(n_vocab: u32, n_audio_ctx: usize) -> Result<Self> {
        if n_vocab <= TIMESTAMP_BEGIN {
            return Err(Error::Tokenizer(format!(
                "standard tokenizer needs a multilingual checkpoint, got n_vocab={n_vocab}"
            )));
        }
        let specials = make_specials(n_vocab);
        let core = Self::standard_core(&specials)?;
        Self::new_from_core(core, n_vocab, n_audio_ctx)
    }

    fn standard_core(specials: &[(String, u32)]) -> Result<CoreBpe> {
        let bytes = include_bytes!("../../assets/multilingual.tiktoken");
        CoreBpe::from_tiktoken_with_special(bytes, specials)
    }

    pub fn new_from_core(core: CoreBpe, n_vocab: u32, n_audio_ctx: usize) -> Result<Self> {
        if n_vocab <= TIMESTAMP_BEGIN {
            return Err(Error::Tokenizer(
                "n_vocab too small for whisper special tokens".into(),
            ));
        }
        let special = special_ids(n_vocab as usize);
        let mut language_ids = HashMap::new();
        for (i, code) in language_codes(special.n_languages).into_iter().enumerate() {
            language_ids.insert(code, LANGUAGE_BASE + i as u32);
        }
        Ok(Self {
            core,
            n_vocab,
            n_audio_ctx,
            timestamp_begin: special.timestamp_begin,
            special,
            language_ids,
        })
    }

    pub fn sot(&self) -> u32 {
        self.special.sot
    }
    pub fn eot(&self) -> u32 {
        self.special.eot
    }
    pub fn translate(&self) -> u32 {
        self.special.translate
    }
    pub fn transcribe(&self) -> u32 {
        self.special.transcribe
    }
    pub fn startoflm(&self) -> u32 {
        self.special.startoflm
    }
    pub fn startofprev(&self) -> u32 {
        self.special.startofprev
    }
    pub fn nospeech(&self) -> u32 {
        self.special.nospeech
    }
    pub fn notimestamps(&self) -> u32 {
        self.special.notimestamps
    }

    pub fn language_token(&self, code: &str) -> Option<u32> {
        self.language_ids.get(code).copied()
    }

    pub fn valid_language_tokens(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.language_ids.values().copied().collect();
        v.sort_unstable();
        v
    }

    pub fn timestamp_token(&self, time: f64) -> u32 {
        let steps = (time.round() / 0.02) as u32;
        self.timestamp_begin + steps
    }

    pub fn timestamp_tokens(&self, start: f64, end: f64) -> Vec<u32> {
        let a = self.timestamp_token(start);
        let b = self.timestamp_token(end);
        let mut out = Vec::new();
        let step: i64 = if b >= a { 1 } else { -1 };
        let mut t = a as i64;
        loop {
            out.push(t as u32);
            if t == b as i64 {
                break;
            }
            t += step;
        }
        out
    }

    pub fn sot_sequence(
        &self,
        lang: Option<&str>,
        task: Task,
        timestamps: bool,
    ) -> Result<Vec<u32>> {
        let mut seq = vec![self.sot()];
        if let Some(code) = lang {
            seq.push(
                self.language_token(code)
                    .ok_or_else(|| Error::Tokenizer(format!("unknown language {code}")))?,
            );
        }
        seq.push(match task {
            Task::Transcribe => self.transcribe(),
            Task::Translate => self.translate(),
        });
        if !timestamps {
            seq.push(self.notimestamps());
        }
        Ok(seq)
    }

    /// Return true if a token is a timestamp token.
    pub fn is_timestamp(&self, token: u32) -> bool {
        token >= self.timestamp_begin
    }

    /// Return true for the non-text special tokens (sot sequence, language /
    /// task markers, eot, nospeech, notimestamps). Timestamps are handled by
    /// [`Self::is_timestamp`]; everything else is decodable text.
    pub fn is_special(&self, token: u32) -> bool {
        let s = self.special;
        token == s.eot
            || token == s.sot
            || token == s.translate
            || token == s.transcribe
            || token == s.startoflm
            || token == s.startofprev
            || token == s.nospeech
            || token == s.notimestamps
            || (LANGUAGE_BASE..LANGUAGE_BASE + s.n_languages as u32).contains(&token)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.core.encode_ordinary(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.core.decode(tokens)
    }
}
