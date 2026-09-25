use crate::{Error, Result};
use base64::Engine as _;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct CoreBpe {
    pub encoder: HashMap<Vec<u8>, u32>,
    pub decoder: Vec<Vec<u8>>,
    pub special_encoder: HashMap<Vec<u8>, u32>,
    pub special_tokens: Vec<(String, u32)>,
}

fn pat_str_re() -> fancy_regex::Regex {
    fancy_regex::Regex::new(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )
    .expect("static regex")
}

impl CoreBpe {
    pub fn from_tiktoken(bytes: &[u8]) -> Result<Self> {
        Self::from_tiktoken_with_special(bytes, &[])
    }

    pub fn from_tiktoken_with_special(bytes: &[u8], specials: &[(String, u32)]) -> Result<Self> {
        use base64::engine::general_purpose::STANDARD as B64;
        let text = std::str::from_utf8(bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut encoder = HashMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (tok, rank) = line
                .rsplit_once(' ')
                .ok_or_else(|| Error::Tokenizer(format!("bad tiktoken line: {line}")))?;
            let rank: u32 = rank
                .trim()
                .parse()
                .map_err(|_| Error::Tokenizer(format!("bad rank in line: {line}")))?;
            let tok: Vec<u8> = if tok.is_empty() || tok.bytes().all(|b| b == b'=') {
                // Reference vocab files reserve rank 50256 for an empty token,
                // encoded as `=` (Python `base64.b64decode("=")` -> b"").
                Vec::new()
            } else {
                B64.decode(tok)
                    .map_err(|e| Error::Tokenizer(format!("base64: {e}")))?
            };
            encoder.insert(tok, rank);
        }
        let mut decoder = vec![Vec::new(); encoder.len()];
        for (k, v) in &encoder {
            decoder[*v as usize] = k.clone();
        }
        let mut special_encoder = HashMap::new();
        for (tok, id) in specials {
            special_encoder.insert(tok.as_bytes().to_vec(), *id);
        }
        Ok(Self {
            encoder,
            decoder,
            special_encoder,
            special_tokens: specials.to_vec(),
        })
    }

    pub fn n_vocab(&self) -> u32 {
        (self.encoder.len() + self.special_encoder.len()) as u32
    }

    pub fn special(&self, token: &str) -> Option<u32> {
        self.special_encoder.get(token.as_bytes()).copied()
    }

    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let re = pat_str_re();
        for m in re
            .find_iter(text)
            .map(|m| m.map_err(|e| Error::Tokenizer(format!("pat_str: {e}"))))
        {
            let p = m?.as_str();
            if let Some(id) = self.special_encoder.get(p.as_bytes()) {
                out.push(*id);
            } else {
                out.extend(self.encode_piece(p)?);
            }
        }
        Ok(out)
    }

    fn encode_piece(&self, piece: &str) -> Result<Vec<u32>> {
        let bytes = piece.as_bytes();
        if bytes.is_empty() {
            return Ok(vec![]);
        }
        if let Some(&id) = self.encoder.get(bytes) {
            return Ok(vec![id]);
        }
        Ok(byte_pair_merge(bytes, &self.encoder))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &t in tokens {
            match self.decoder.get(t as usize) {
                Some(b) => bytes.extend_from_slice(b),
                None => {
                    if let Some((tk, _)) = self.special_tokens.iter().find(|(_, id)| *id == t) {
                        bytes.extend_from_slice(tk.as_bytes());
                    } else {
                        return Err(Error::Tokenizer(format!("decode: unknown token id {t}")));
                    }
                }
            }
        }
        String::from_utf8(bytes).map_err(|e| Error::Tokenizer(format!("utf8: {e}")))
    }
}

fn byte_pair_merge(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.len() == 1 {
        return match ranks.get(piece) {
            Some(&r) => vec![r],
            None => vec![0],
        };
    }
    let rank_of_parts = |a: (usize, usize), b: (usize, usize)| -> Option<u32> {
        let mut key = Vec::with_capacity(a.1 + b.1);
        key.extend_from_slice(&piece[a.0..a.0 + a.1]);
        key.extend_from_slice(&piece[b.0..b.0 + b.1]);
        ranks.get(&key).copied()
    };
    let mut starts: Vec<usize> = (0..piece.len()).collect();
    let mut lens: Vec<usize> = vec![1; piece.len()];
    let mut parts_rank: Vec<Option<u32>> = (0..starts.len().saturating_sub(1))
        .map(|i| rank_of_parts((starts[i], lens[i]), (starts[i + 1], lens[i + 1])))
        .collect();

    while !parts_rank.is_empty() {
        let best = parts_rank
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .min_by_key(|(_, r)| *r);
        let i = match best {
            Some((i, _)) => i,
            None => break,
        };
        lens[i] += lens[i + 1];
        starts.remove(i + 1);
        lens.remove(i + 1);
        parts_rank.remove(i);
        if i < parts_rank.len() {
            parts_rank[i] = rank_of_parts((starts[i], lens[i]), (starts[i + 1], lens[i + 1]));
        }
        if i > 0 {
            parts_rank[i - 1] = rank_of_parts((starts[i - 1], lens[i - 1]), (starts[i], lens[i]));
        }
    }

    starts
        .iter()
        .zip(lens.iter())
        .map(|(&s, &l)| {
            let key = piece[s..s + l].to_vec();
            ranks
                .get(&key)
                .copied()
                .unwrap_or_else(|| ranks.get(&key[..1]).copied().unwrap_or(0))
        })
        .collect()
}
