import { useEffect, useRef } from "react";
import type { ProgressDto } from "../types";

export function StatusPanel({ progress }: { progress: ProgressDto[] }) {
  const listRef = useRef<HTMLUListElement>(null);

  const lastDownload = [...progress].reverse().find((p) => p.kind === "download");
  const fraction =
    lastDownload && lastDownload.fraction != null ? lastDownload.fraction : null;

  useEffect(() => {
    listRef.current?.lastElementChild?.scrollIntoView({ block: "end" });
  }, [progress]);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h2 className="mb-2 text-sm font-semibold text-slate-500 dark:text-slate-400">
        Status
      </h2>
      {progress.length === 0 && (
        <p className="text-sm text-slate-400">Pick an audio file and press Transcribe.</p>
      )}
      {fraction != null && (
        <div className="mb-3 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
          <div
            className="h-full bg-blue-600 transition-all"
            style={{ width: `${Math.round(fraction * 100)}%` }}
          />
        </div>
      )}
      <div className="h-40 overflow-y-auto">
        <ul ref={listRef} className="space-y-1 text-sm">
          {progress.slice(-50).map((p, i) => (
            <li key={i} className="flex gap-2 text-slate-600 dark:text-slate-300">
              <span className="shrink-0 rounded bg-slate-100 px-1.5 text-xs uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                {p.kind}
              </span>
              <span className="truncate font-mono">{p.message}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
