# Transcriptor

Transcriptor is a Rust + React app that transcribes WAV audio with Whisper running on the Candle crate.

## Stack

- **Backend**: Rust, `actix-web`, Candle + `candle-transformers` Whisper
- **Frontend**: React (Vite)

## Features

- Upload a WAV audio file from the React UI
- Run transcription on the Rust backend using Whisper (tiny.en)
- Download transcription as **TXT** or **DOCX**

## Run backend

```bash
cargo run
```

Backend starts on `http://localhost:8080`.

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend starts on `http://localhost:5173` and calls `http://localhost:8080` by default.

To customize backend URL:

```bash
VITE_API_BASE_URL=http://localhost:8080 npm run dev
```

## Notes

- Whisper model files are fetched from Hugging Face on first transcription request and cached locally by `hf-hub`.
- Current backend expects **16kHz WAV** input to match Whisper preprocessing.
