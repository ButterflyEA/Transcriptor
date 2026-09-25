"""One-time dev tool: emit T20 golden values (mel + first encoder pass).

Builds a 30 s window that holds a 1 s, 440 Hz tone (0.2 amplitude @ 16 kHz,
zero-padded like `features::window::split_chunks`), then computes:
  * reference log-mel via openai-whisper (`log_mel_spectrogram`),
  * reference encoder output on that mel via the reference tiny model,
and writes the selected columns / head rows to tests/fixtures/golden_tone.json.

Usage:  python scripts/generate_encoder_golden.py
Requires: openai-whisper + torch (CPU is fine).
"""

import json
import math
import pathlib

import numpy as np
import torch
import whisper

SR = 16000
N_SAMPLES = 480_000  # 30 s

tone = [0.2 * math.sin(2 * math.pi * 440.0 * i / SR) for i in range(16000)]
audio = np.zeros(N_SAMPLES, dtype=np.float32)
audio[:16000] = tone

mel = whisper.log_mel_spectrogram(audio)  # [80, 3000]

model = whisper.load_model("tiny")
with torch.no_grad():
    enc = model.encoder(mel.unsqueeze(0))  # [1, 1500, 384]

mcols = [0, 5, 49, 50, 99]
erows = [0, 749, 1499]

out = {
    "mel_shape": list(mel.shape),
    "enc_shape": list(enc.shape),
    "mel_cols": {
        str(c): [round(float(v), 6) for v in mel[:, c].tolist()] for c in mcols
    },
    "enc_head": {
        str(r): [round(float(v), 6) for v in enc[0, r, :8].tolist()] for r in erows
    },
}

path = pathlib.Path("crates/whisper-burn/tests/fixtures/golden_tone.json")
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", path, "(mel", out["mel_shape"], "enc", out["enc_shape"], ")")