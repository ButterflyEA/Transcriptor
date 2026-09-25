# whisper-burn

A from-scratch Rust implementation of [OpenAI Whisper](https://github.com/openai/whisper)
running on [burn](https://burn.dev) — a Cargo workspace with two crates:

- **`whisper-burn`** — the library: model definition, GGML weight loading,
  tokenizer/BPE, log-mel, greedy & beam decoding, and full transcription
  orchestration.
- **`whisper-burn-cli`** — a thin command-line frontend that transcribes
  audio files on a wgpu (GPU) or ndarray (CPU) backend.

## Transcribe from the command line

```sh
cargo run --release -p whisper-burn-cli -- speech.wav
```

Prints timestamped segments to stdout and can write `txt`, `srt`, `vtt`, and
`json` transcripts next to your audio file:

```sh
cargo run --release -p whisper-burn-cli -- interview.flac --model base --output srt
```

For everything else — options, examples, output formats, model downloads, and
troubleshooting — see [**docs/usage.md**](docs/usage.md).

## Tests

```sh
cargo test --workspace                # offline suite
cargo test --workspace -- --ignored   # + network integration tests (real audio, model weights)
```