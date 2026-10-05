//! SHA-256 helpers for `dictation verify-model`: lets IT confirm that a model
//! copied into an airgapped network arrived intact.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Parses `sha256sum` output: `<hex>  <name>` (a leading `*` on the name marks binary mode).
pub fn parse_sums(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let (hash, name) = l.trim().split_once(char::is_whitespace)?;
            let name = name.trim().trim_start_matches('*');
            (hash.len() == 64 && !name.is_empty())
                .then(|| (hash.to_ascii_lowercase(), name.to_string()))
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub enum FileCheck {
    Ok,
    Missing,
    Mismatch { actual: String },
}

/// Checks every file listed in `SHA256SUMS` inside `dir`.
pub fn verify_dir(dir: &Path) -> Result<Vec<(String, FileCheck)>, String> {
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS"))
        .map_err(|e| format!("cannot read {}: {e}", dir.join("SHA256SUMS").display()))?;
    let entries = parse_sums(&sums);
    if entries.is_empty() {
        return Err("SHA256SUMS lists no files".into());
    }
    Ok(entries
        .into_iter()
        .map(|(want, name)| {
            let path = dir.join(&name);
            let check = if !path.is_file() {
                FileCheck::Missing
            } else {
                match sha256_file(&path) {
                    Ok(actual) if actual == want => FileCheck::Ok,
                    Ok(actual) => FileCheck::Mismatch { actual },
                    Err(_) => FileCheck::Missing,
                }
            };
            (name, check)
        })
        .collect())
}
