"""One-time dev tool: extract Whisper's mel filters to Rust assets.

Usage:  python scripts/extract_mel_filters.py
Requires numpy. Reads the official mel_filters.npz (download it from the
openai/whisper repo root: whisper/assets/mel_filters.npz) and writes
little-endian f32 row-major arrays, one filter per row.
"""
import numpy as np, pathlib, urllib.request

URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
outdir = pathlib.Path("assets")
outdir.mkdir(parents=True, exist_ok=True)

tmp = pathlib.Path("target/mel_filters.npz")
if not tmp.exists():
    tmp.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, tmp)

data = np.load(tmp)
for n in (80, 128):
    arr = data[f"mel_{n}"]          # shape (n_mels, n_fft//2+1) in the npz
    arr = np.ascontiguousarray(arr, dtype=np.float32)   # (n_mels, 201)
    arr.tofile(outdir / f"mel_filters_{n}.bin")
    print(n, arr.shape)
# Print a provenance checksum of the first row for later cement
print("first row of mel_80:", data["mel_80"][0, :6])