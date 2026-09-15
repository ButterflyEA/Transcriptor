import { useMemo, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

function App() {
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [filename, setFilename] = useState('transcription')

  const hasText = useMemo(() => text.trim().length > 0, [text])

  const onSubmit = async (event) => {
    event.preventDefault()
    if (!file) {
      setError('Please choose a WAV audio file first.')
      return
    }

    setIsLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('audio', file)

      const response = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Transcription failed')
      }

      const data = await response.json()
      setText(data.text ?? '')
      if (!filename.trim()) {
        setFilename(file.name.replace(/\.[^/.]+$/, '') || 'transcription')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const download = async (format) => {
    setError('')
    const response = await fetch(`${API_BASE_URL}/download/${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, filename }),
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || `Failed to download ${format}`)
    }

    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${filename || 'transcription'}.${format}`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="app">
      <h1>Transcriptor</h1>
      <p>Upload WAV audio, transcribe with Whisper (Candle), then download TXT or DOCX.</p>

      <form onSubmit={onSubmit} className="panel">
        <label htmlFor="audio">Audio file (WAV, 16kHz)</label>
        <input
          id="audio"
          type="file"
          accept="audio/wav,.wav"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />

        <label htmlFor="filename">Download filename</label>
        <input
          id="filename"
          value={filename}
          onChange={(event) => setFilename(event.target.value)}
          placeholder="transcription"
        />

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Transcribing…' : 'Run transcription'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      <section className="panel">
        <h2>Transcription</h2>
        <textarea value={text} onChange={(event) => setText(event.target.value)} rows={10} />
        <div className="actions">
          <button type="button" disabled={!hasText} onClick={() => download('txt').catch((err) => setError(err.message))}>
            Download TXT
          </button>
          <button type="button" disabled={!hasText} onClick={() => download('docx').catch((err) => setError(err.message))}>
            Download DOCX
          </button>
        </div>
      </section>
    </main>
  )
}

export default App
