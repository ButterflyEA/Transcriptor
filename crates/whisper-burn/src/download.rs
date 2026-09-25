use crate::config::ModelSize;
use crate::weights::safetensors::merge_safetensors;
use crate::{Error, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const HF_BASE: &str = "https://huggingface.co";

const MAX_ATTEMPTS: usize = 3;

/// Shared progress sink for downloads: reports `downloaded` bytes and the
/// `total` when `Content-Length` is available (else `None`).
pub type DownloadCallback = Arc<dyn Fn(u64, Option<u64>) + Send + Sync>;

/// Hugging Face resolve URL for a model size, e.g.
/// `https://huggingface.co/openai/whisper-tiny/resolve/main`.
pub fn repo_base(size: ModelSize) -> String {
    format!("{HF_BASE}/{}/resolve/main", size.repo_id())
}

/// Downloads `url` into `dest`, writing atomically (`.part` then rename).
/// Retries transport errors up to `MAX_ATTEMPTS`; HTTP statuses fail fast.
/// When `overwrite` is false and `dest` already exists it is left untouched.
pub fn download_to(url: &str, dest: &Path, overwrite: bool) -> Result<PathBuf> {
    download_to_with_progress(url, dest, overwrite, None)
}

/// Like [`download_to`] but reports progress through `on_progress`
/// (`downloaded` bytes, `total` when `Content-Length` is available).
pub fn download_to_with_progress(
    url: &str,
    dest: &Path,
    overwrite: bool,
    on_progress: Option<DownloadCallback>,
) -> Result<PathBuf> {
    if !overwrite && dest.exists() {
        return Ok(dest.to_path_buf());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(dest);
    match fetch_with_retry(url, &part, &on_progress) {
        Ok(()) => {
            std::fs::rename(&part, dest)?;
            Ok(dest.to_path_buf())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

fn fetch_with_retry(url: &str, part: &Path, on_progress: &Option<DownloadCallback>) -> Result<()> {
    for attempt in 0..MAX_ATTEMPTS {
        let last = match attempt_download(url, part, on_progress) {
            Ok(()) => return Ok(()),
            Err(fail) => {
                let _ = std::fs::remove_file(part);
                match fail.kind {
                    DownloadFailKind::Status => return Err(Error::Download(fail.message)),
                    DownloadFailKind::Transport => fail.message,
                }
            }
        };
        if attempt + 1 == MAX_ATTEMPTS {
            return Err(Error::Download(format!(
                "after {MAX_ATTEMPTS} attempts: {last}"
            )));
        }
    }
    unreachable!()
}

#[derive(Debug, PartialEq)]
enum DownloadFailKind {
    Status,
    Transport,
}

struct DownloadFail {
    kind: DownloadFailKind,
    message: String,
}

impl std::fmt::Display for DownloadFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn attempt_download(
    url: &str,
    part: &Path,
    on_progress: &Option<DownloadCallback>,
) -> std::result::Result<(), DownloadFail> {
    let response = match ureq::get(url).call() {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            return Err(DownloadFail {
                kind: DownloadFailKind::Status,
                message: format!("HTTP {code} for {url} ({})", response.status_text()),
            });
        }
        Err(e) => {
            return Err(DownloadFail {
                kind: DownloadFailKind::Transport,
                message: e.to_string(),
            });
        }
    };
    let total = response
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut file = File::create(part).map_err(|e| DownloadFail {
        kind: DownloadFailKind::Transport,
        message: e.to_string(),
    })?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadFail {
            kind: DownloadFailKind::Transport,
            message: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| DownloadFail {
            kind: DownloadFailKind::Transport,
            message: e.to_string(),
        })?;
        downloaded += n as u64;
        if let Some(cb) = on_progress {
            cb(downloaded, total);
        }
    }
    file.flush().map_err(|e| DownloadFail {
        kind: DownloadFailKind::Transport,
        message: e.to_string(),
    })?;
    Ok(())
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: std::collections::HashMap<String, String>,
}

/// Extracts the shard file names from a `model.safetensors.index.json` body,
/// sorted (HF shard names sort by part number) and deduplicated.
pub fn shard_filenames(index: &str) -> Result<Vec<String>> {
    let index: IndexFile = serde_json::from_str(index)
        .map_err(|e| Error::Download(format!("model.safetensors.index.json: {e}")))?;
    let mut names: Vec<String> = index.weight_map.into_values().collect();
    names.sort();
    names.dedup();
    Ok(names)
}

/// Downloads `config.json` (always) and `model.safetensors` (when
/// `require_weights`) into `dest_dir`, returning the safetensors path when
/// weights were fetched, otherwise the config path. Sharded checkpoints
/// (ivrit-ai) are downloaded as their per-part files and merged into a single
/// `model.safetensors`.
pub fn download_checkpoint(
    size: ModelSize,
    dest_dir: &Path,
    overwrite: bool,
    require_weights: bool,
) -> Result<PathBuf> {
    download_checkpoint_with_progress(size, dest_dir, overwrite, require_weights, None)
}

/// [`download_checkpoint`] that also reports per-file download progress via
/// `on_progress` (bytes downloaded, total from `Content-Length` when known).
pub fn download_checkpoint_with_progress(
    size: ModelSize,
    dest_dir: &Path,
    overwrite: bool,
    require_weights: bool,
    on_progress: Option<DownloadCallback>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let base = repo_base(size);
    log::info!("weights: fetching config.json from {base}");
    let config_path = download_to_with_progress(
        &format!("{base}/config.json"),
        &dest_dir.join("config.json"),
        overwrite,
        None,
    )?;
    if !require_weights {
        return Ok(config_path);
    }
    if size.is_sharded() {
        return download_sharded_checkpoint(&base, dest_dir, overwrite, &on_progress);
    }
    log::info!("weights: fetching model.safetensors from {base}");
    let weights_path = download_to_with_progress(
        &format!("{base}/model.safetensors"),
        &dest_dir.join("model.safetensors"),
        overwrite,
        on_progress,
    )?;
    Ok(weights_path)
}

/// Fetches each shard listed in `model.safetensors.index.json`, merges them
/// into `model.safetensors`, and removes the transient part/index files.
fn download_sharded_checkpoint(
    base: &str,
    dest_dir: &Path,
    overwrite: bool,
    on_progress: &Option<DownloadCallback>,
) -> Result<PathBuf> {
    let st_path = dest_dir.join("model.safetensors");
    if !overwrite && st_path.exists() {
        return Ok(st_path);
    }
    let index_path = dest_dir.join("model.safetensors.index.json");
    let downloaded = download_to(
        &format!("{base}/model.safetensors.index.json"),
        &index_path,
        overwrite,
    )?;
    let index = std::fs::read_to_string(&downloaded).map_err(|source| Error::Io {
        path: downloaded,
        source,
    })?;
    let shard_names = shard_filenames(&index)?;
    let mut shards = Vec::with_capacity(shard_names.len());
    for name in &shard_names {
        let part = dest_dir.join(name);
        log::info!("weights: fetching {name} from {base}");
        let path = download_to_with_progress(
            &format!("{base}/{name}"),
            &part,
            overwrite,
            on_progress.clone(),
        )?;
        let bytes = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
        log::info!(
            "weights: {name} ({:.0} MB)",
            bytes.len() as f64 / (1024.0 * 1024.0)
        );
        shards.push(bytes);
    }
    log::info!(
        "weights: merging {} shards into model.safetensors",
        shards.len()
    );
    let merged = merge_safetensors(&shards)?;
    drop(shards);
    let tmp = part_path(&st_path);
    std::fs::write(&tmp, merged).map_err(|source| Error::Io {
        path: tmp.clone(),
        source,
    })?;
    if st_path.exists() {
        std::fs::remove_file(&st_path).map_err(|source| Error::Io {
            path: st_path.clone(),
            source,
        })?;
    }
    std::fs::rename(&tmp, &st_path).map_err(|source| Error::Io {
        path: st_path.clone(),
        source,
    })?;
    for name in &shard_names {
        let _ = std::fs::remove_file(dest_dir.join(name));
    }
    let _ = std::fs::remove_file(&index_path);
    Ok(st_path)
}

/// Cache directory for a model: `WHISPER_BURN_CACHE` if set, otherwise
/// `~/.cache/whisper-burn/{repo_id}`.
pub fn cache_dir(size: ModelSize) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("WHISPER_BURN_CACHE") {
        return Ok(PathBuf::from(p).join(size.repo_id()));
    }
    let home = home_dir().ok_or_else(|| Error::Io {
        path: PathBuf::new(),
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory for model cache",
        ),
    })?;
    Ok(home
        .join(".cache")
        .join("whisper-burn")
        .join(size.repo_id()))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_filenames_are_sorted_and_deduped() {
        let index = r#"{"metadata":{"total_size":1000},"weight_map":{
            "model.a":"model-00002-of-00002.safetensors",
            "model.b":"model-00001-of-00002.safetensors",
            "model.c":"model-00001-of-00002.safetensors"
        }}"#;
        assert_eq!(
            shard_filenames(index).unwrap(),
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
    }

    #[test]
    fn malformed_index_errors() {
        assert!(shard_filenames("not json").is_err());
    }
}
