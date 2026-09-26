#!/usr/bin/env python3
"""Regenerate the golden transcripts from the *official* reference whisper.

The goldens in docs/golden/ are ground truth: they are what openai/whisper
itself produces, so comparing against them is the only honest check that our
Rust port behaves like the original. Everything else is self-consistency.

The reference decides segment boundaries, so the emitted JSON uses exactly the
schema Transcriptor produces:

    {"language": "he", "segments": [{"start": "00:00.000", "end": "07.800", ...}]}

Usage:
    python scripts/cement_reference.py --model large-v3 --language auto
    python scripts/cement_reference.py --model tiny --language en

Omit --language (or pass "auto") to let the reference auto-detect.

Why an explicit --language: auto-detection and forced transcription are
different code paths in the reference and produce genuinely different output.
Both are worth pinning, but they must be pinned as separate goldens.

Requires the reference implementation, NOT this repo:
    pip install openai-whisper torch
and a Python that torch ships wheels for (cp311 is safe; cp314 is not).
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "docs" / "golden"

# Transcribed with the reference's own search settings so the goldens describe
# a reproducible run rather than whatever the defaults drift into.
#
#   temperature=0.0 disables openai/whisper's temperature fallback chain. The
#   fallback re-decodes on low confidence and is the single biggest source of
#   run-to-run variation; our port has no equivalent, so leaving it on would
#   compare two different algorithms. The remaining thresholds are the
#   library's own defaults and are passed explicitly so a future library
#   default change cannot silently invalidate the goldens.
BEAM_SIZE = 5
TEMPERATURE = 0.0
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6
COMPRESSION_RATIO_THRESHOLD = 2.4
CONDITION_ON_PREVIOUS_TEXT = True

# Our decoder works in 20 ms timestamp steps, the same granularity the
# reference uses when it rounds segment times to 2 decimals. Timestamps are
# emitted as MM:SS.mmm with unbounded minutes to match format_json().
_CLOCK_ROUNDING = 2


def load_wav_mono_f32(path: Path) -> np.ndarray:
    """Read a 16-bit PCM wav as mono float32 in [-1, 1].

    Bypasses ffmpeg on purpose: openai/whisper shells out to ffmpeg when given
    a filename, and requiring a binary that is not always installed makes the
    goldens unreproducible. Dividing by 32768 matches ffmpeg's own s16
    conversion, so the samples are identical either way.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {wav.getsampwidth() * 8}-bit")
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def to_clock(seconds: float) -> str:
    """Seconds -> MM:SS.mmm, the exact shape format_json() emits."""
    ms = int(round(seconds * 1000))
    secs, millis = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


def normalise_text(text: str, keep_leading_space: bool) -> str:
    """Apply Transcriptor's leading-space convention to reference text.

    Official whisper emits every segment's text with a leading space, because
    the BPE word-start marker survives into the decoded string. Transcriptor
    strips it: the timestamp already supplies the word break, and a stray
    space is ugly in .txt/.srt exports and in concatenated transcripts.

    The golden is stored in the *same* normalised form the app emits, so the
    Rust comparison stays a plain byte-for-byte check with no normalisation
    hidden in the test. The stripping itself is pinned by its own unit test
    (segment_text_drops_the_post_timestamp_space), so it cannot rot.

    Only the leading space is touched. Trailing whitespace is left alone: the
    reference does not emit any, and trimming it would mask a real difference.
    """
    return text if keep_leading_space else text.lstrip(" ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo")
    parser.add_argument("--language", default="auto", help="ISO code, or 'auto' to detect")
    parser.add_argument("--audio", default=str(REPO / "test_16000_mono.wav"))
    parser.add_argument("--out-dir", default=str(GOLDEN_DIR))
    parser.add_argument("--stdout", action="store_true", help="print the golden instead of writing it")
    parser.add_argument(
        "--keep-leading-space",
        action="store_true",
        help="store the reference's raw text, leading space included, instead of Transcriptor's stripped form",
    )
    args = parser.parse_args()

    try:
        import torch
        import whisper
    except ImportError as exc:  # pragma: no cover - environment guard
        print(f"the reference implementation is required: {exc}", file=sys.stderr)
        print("pip install openai-whisper torch", file=sys.stderr)
        return 2

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"no such audio file: {audio_path}", file=sys.stderr)
        return 2

    language = None if args.language == "auto" else args.language
    audio = load_wav_mono_f32(audio_path)
    duration = len(audio) / 16000.0

    print(
        f"reference whisper | model={args.model} language={args.language} "
        f"beam={BEAM_SIZE} temperature={TEMPERATURE} cuda={torch.cuda.is_available()}",
        file=sys.stderr,
    )
    print(f"audio: {duration:.3f}s @ 16 kHz mono", file=sys.stderr)

    model = whisper.load_model(args.model)
    result: dict[str, Any] = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        logprob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        initial_prompt=None,
        word_timestamps=False,
        fp16=False,
        verbose=None,
    )
    segments: list[dict[str, Any]] = result["segments"]

    # The reference rounds to 2 decimals itself; rounding again is a no-op that
    # documents the granularity our 20 ms timestamp steps must reproduce.
    golden = {
        "language": result["language"],
        "segments": [
            {
                "start": to_clock(round(seg["start"], _CLOCK_ROUNDING)),
                "end": to_clock(round(seg["end"], _CLOCK_ROUNDING)),
                "text": normalise_text(seg["text"], args.keep_leading_space),
            }
            for seg in segments
        ],
    }

    payload = json.dumps(golden, ensure_ascii=False, indent=2) + "\n"

    if args.stdout:
        sys.stdout.write(payload)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "auto" if args.language == "auto" else args.language
    out_path = out_dir / f"reference_{args.model}_{suffix}_golden.json"
    out_path.write_text(payload, encoding="utf-8")

    # Provenance travels with the golden. A reference file with no recorded
    # library version is a liability: it cannot be reproduced or invalidated.
    provenance = out_dir / f"reference_{args.model}_{suffix}.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "model": args.model,
                "requested_language": args.language,
                "resolved_language": golden["language"],
                "beam_size": BEAM_SIZE,
                "temperature": TEMPERATURE,
                "logprob_threshold": LOGPROB_THRESHOLD,
                "no_speech_threshold": NO_SPEECH_THRESHOLD,
                "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
                "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
                "audio": audio_path.name,
                "audio_duration_s": round(duration, 6),
                "segments": len(golden["segments"]),
                "leading_space_stripped": not args.keep_leading_space,
                "torch": torch.__version__,
                "whisper": getattr(whisper, "__version__", "unknown"),
                "python": sys.version.split()[0],
                "cuda": bool(torch.cuda.is_available()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_path} ({len(golden['segments'])} segments)", file=sys.stderr)
    print(f"wrote {provenance}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
