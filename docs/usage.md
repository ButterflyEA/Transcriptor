# whisper-burn — CLI usage

`whisper-burn-cli` transcribes audio to timestamped text using OpenAI
Whisper models running on [burn](https://burn.dev) backends (GPU via wgpu,
pure-CPU via ndarray).

## Building

Requires a [Rust toolchain](https://rustup.rs) (stable). From the workspace
root:

```sh
cargo build --release -p whisper-burn-cli
```

The binary lands at `target/release/whisper-burn-cli` (add `.exe` on
Windows). For everyday use, put it on your PATH:

```sh
cargo install --path crates/whisper-burn-cli
```

or run it directly with Cargo:

```sh
cargo run --release -p whisper-burn-cli -- <audio>
```

## Quick start

```sh
# transcribe a file with the default tiny model
whisper-burn-cli audio.wav

# transcribe a longer clip with a bigger model and write an srt file
whisper-burn-cli speech.flac --model base --output srt

# force CPU (no GPU) and auto-detect the language
whisper-burn-cli interview.mp3 --device cpu
```

Output is printed to stdout as console-style segments:

```
[00:00.000 --> 00:30.000]  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
```

When `--output` is given, an additional transcript file is written next to
the audio file (`audio.txt`, `audio.srt`, `audio.vtt`, or `audio.json`).

## Options

| Option | Default | Description |
| --- | --- | --- |
| `<AUDIO>` | — | Path to an audio file (wav, flac, mp3, ...). |
| `--model <SIZE>` | `tiny` | Model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo`. Larger models are slower but more accurate. |
| `--language <CODE>` | auto | Spoken language code (e.g. `en`, `es`, `ja`). When omitted it is detected automatically from the first seconds of audio. |
| `--task <TASK>` | `transcribe` | `transcribe` (same language) or `translate` (to English). |
| `--beam-size <N>` | `0` | Beam search width. `0` selects greedy decoding. |
| `--output <FMT>` | — | Also write the transcript file: `txt`, `srt`, `vtt`, or `json`. |
| `--device <DEV>` | `wgpu` | Backend device. `wgpu` uses a GPU if one is available and falls back to CPU automatically; `cpu` forces the ndarray CPU backend. |
| `--verbose` | off | Print diagnostics (model, device, decode stats) to stderr. |
| `--progress` | off | Stream pipeline stage progress (model loading, audio conversion, language detection, per-window decode, finished segments) as `[elapsed]` lines to stderr. The transcript stays on stdout. |
| `--initial-prompt <TEXT>` | — | Text conditioning the first decoding window (reference whisper `initial_prompt`). |
| `-h` / `-V` | — | Help / version. |

## Examples

Detect the language and translate a French clip to English with beam search:

```sh
whisper-burn-cli le_discours.flac --task translate --beam-size 4
```

Write a mixed set of transcripts next to the file:

```sh
whisper-burn-cli podcast.flac --model small --output json
whisper-burn-cli podcast.flac --model small --output srt
```

Inspect what the CLI is doing:

```sh
whisper-burn-cli audio.mp3 --verbose --output vtt
```

Watch each pipeline stage and the decoded segments as they finish:

```sh
whisper-burn-cli audio.mp3 --model base --progress
```

The progress lines look like (written to stderr, transcript unaffected):

```
[   0.3s] weights: openai/whisper-base cached at C:\Users\me\.cache\whisper-burn\openai/whisper-base
[   0.3s] weights: reading C:\Users\me\.cache\whisper-burn\openai/whisper-base\model.safetensors (142 MB)
[   0.4s] model: openai/whisper-base (12 blocks, n_mels 80, n_ctx 448)
[   0.5s] decode: 22.1 s of audio in 1 window(s) of 30 s
[   0.6s] decode: window 1/1 (00:00.000 – 00:30.000)
[   4.0s] decode: detected language en
[   7.0s] decode: window 1/1: language=en beam=0 tokens=45 no-speech=0.000 mean-logprob=-0.231
[   7.0s]   [00:00.000 --> 00:02.000]   And so my fellow Americans
```

## Output formats

- `txt` — one line of text per segment.
- `srt` — SubRip cues (`00:00:00,000 --> 00:00:00,400`).
- `vtt` — WebVTT with a `WEBVTT` header.
- `json` — array of `{ "start", "end", "text" }` objects, times in seconds:

```json
[{"end":30.0,"start":0.0,"text":" And so my fellow Americans ..."}]
```

## Model downloads

Weights are downloaded on first use from
Hugging Face (`openai/whisper-*`) and cached under
`~/.cache/whisper-burn/{repo_id}` — on Windows this is
`%USERPROFILE%\.cache\whisper-burn\openai\whisper-tiny`,
etc. Point `WHISPER_BURN_CACHE` at another directory to relocate the cache:

```sh
export WHISPER_BURN_CACHE=/mnt/models   # POSIX
set WHISPER_BURN_CACHE=C:\models        # Windows cmd
```

The first run of a new model size downloads several hundred MB; the default
`tiny` model is ~72 MB.

## Troubleshooting

- **"no usable wgpu adapter"** — no GPU was found; the CLI falls back to the
  CPU backend automatically. Force it explicitly with `--device cpu`.
- **Model download is slow** — only the first run downloads; you can
  pre-seed the cache from another machine with the same `repo_id` folder.
- **Which model should I use?** Start with `tiny` (fast, low resource). Move
  up to `base`/`small` for better accuracy on your hardware budget.