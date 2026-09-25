import { useCallback, useState } from "react";
import { clock, type SegmentDto } from "../types";

function renderText(segments: SegmentDto[], timestamps: boolean): string {
  return segments
    .map((s) =>
      timestamps ? `[${clock(s.start)} --> ${clock(s.end)}]  ${s.text}` : s.text,
    )
    .join("\n");
}

interface Props {
  segments: SegmentDto[];
  running: boolean;
  onSave: (format: string, timestamps: boolean) => void;
}

export function TranscriptPanel({ segments, running, onSave }: Props) {
  const [timestamps, setTimestamps] = useState(true);
  const [format, setFormat] = useState("txt");

  const copy = useCallback(async () => {
    const text = renderText(segments, timestamps);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  }, [segments, timestamps]);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400">
          Transcript {running && segments.length > 0 ? `(${segments.length} segments…)` : ""}
        </h2>
        <div className="flex items-center gap-2 text-sm">
          <label className="flex items-center gap-1 text-slate-600 dark:text-slate-300">
            <input
              type="checkbox"
              checked={timestamps}
              onChange={(e) => setTimestamps(e.target.checked)}
            />
            timestamps
          </label>
          <button
            onClick={copy}
            disabled={segments.length === 0}
            className="rounded-lg border border-slate-300 px-2.5 py-1 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Copy
          </button>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
          >
            {["txt", "srt", "vtt", "json", "docx"].map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
          <button
            onClick={() => onSave(format, timestamps)}
            disabled={segments.length === 0}
            className="rounded-lg bg-blue-600 px-3 py-1 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
      <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap p-4 font-mono text-sm leading-relaxed">
        {segments.length === 0
          ? running
            ? "Transcribing…"
            : "Transcript will appear here."
          : renderText(segments, timestamps)}
      </pre>
    </section>
  );
}