import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  AudioInfo,
  Defaults,
  DoneDto,
  ProgressDto,
  SegmentDto,
  TaskParams,
} from "./types";
import { ParamsRail } from "./components/ParamsRail";
import { StatusPanel } from "./components/StatusPanel";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { AdBanner } from "./components/AdBanner";

const EXT: Record<string, string> = {
  txt: "txt", srt: "srt", vtt: "vtt", json: "json", docx: "docx",
};

const errMsg = (e: unknown): string => {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  if (e && typeof e === "object" && "message" in e) {
    const m = (e as { message: unknown }).message;
    if (typeof m === "string") return m;
  }
  return JSON.stringify(e);
};

export default function App() {
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [params, setParams] = useState<TaskParams>({
    model: "tiny", language: "auto", task: "transcribe",
    beamSize: 0, initialPrompt: "", device: "wgpu",
  });
  const [audio, setAudio] = useState<AudioInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressDto[]>([]);
  const [segments, setSegments] = useState<SegmentDto[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  const segmentsRef = useRef<SegmentDto[]>([]);
  segmentsRef.current = segments;

  useEffect(() => {
    invoke<Defaults>("get_defaults")
      .then((d) => {
        setDefaults(d);
        setParams((p) => ({
          ...p, model: d.defaultModel, task: d.defaultTask as TaskParams["task"],
          beamSize: d.defaultBeamSize,
        }));
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    const p = listen<ProgressDto>("progress", (e) =>
      setProgress((prev) => [...prev.slice(-199), e.payload]),
    );
    const s = listen<SegmentDto>("segment", (e) =>
      setSegments((prev) => [...prev, e.payload]),
    );
    const d = listen<DoneDto>("done", (e) => {
      setRunning(false);
      if (e.payload.status === "error") {
        setError(e.payload.message);
        setSegments([]);
      } else {
        setError(null);
        setSegments(e.payload.segments);
      }
    });
    return () => {
      void p.then((f) => f());
      void s.then((f) => f());
      void d.then((f) => f());
    };
  }, []);

  const browse = useCallback(async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"] }],
    });
    if (typeof file !== "string") return;
    try {
      const info = await invoke<AudioInfo>("inspect_audio", { path: file });
      setAudio(info);
      setSegments([]);
      setProgress([]);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const run = useCallback(async () => {
    if (!audio || running) return;
    setError(null);
    setProgress([]);
    setSegments([]);
    setRunning(true);
    try {
      await invoke("transcribe", { path: audio.path, params });
    } catch (e) {
      setRunning(false);
      setError(String(e));
    }
  }, [audio, running, params]);

  const stop = useCallback(async () => {
    try { await invoke("stop"); } catch (e) { setError(String(e)); }
  }, []);

  const saveAs = useCallback(async (format: string, includeTimestamps: boolean) => {
    const target = await save({
      defaultPath: `transcript.${EXT[format] ?? format}`,
      filters: [{ name: format.toUpperCase(), extensions: [EXT[format] ?? format] }],
    });
    if (!target) return;
    try {
      await invoke("save_transcript", {
        request: { path: target, format, includeTimestamps },
      });
    } catch (e) {
      setError(String(e));
    }
  }, []);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3 dark:border-slate-800 dark:bg-slate-900">
        <div>
          <h1 className="text-lg font-semibold">Transcriptor</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Whisper on burn — transcribe or translate audio
          </p>
        </div>
        <button
          onClick={() => setDark((v) => !v)}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          {dark ? "Light" : "Dark"}
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <ParamsRail
          defaults={defaults}
          params={params}
          onChange={setParams}
          audio={audio}
          running={running}
          onBrowse={browse}
          onRun={run}
          onStop={stop}
        />
        <main className="flex min-h-0 flex-1 flex-col gap-3 p-4">
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
              {error}
            </div>
          )}
          <StatusPanel progress={progress} />
          <TranscriptPanel segments={segments} running={running} onSave={saveAs} />
        </main>
      </div>

      <AdBanner text="Transcriptor made by ButterflyEA" />
    </div>
  );
}