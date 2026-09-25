import type { AudioInfo, Defaults, TaskParams } from "../types";

interface Props {
  defaults: Defaults | null;
  params: TaskParams;
  onChange: (p: TaskParams) => void;
  audio: AudioInfo | null;
  running: boolean;
  onBrowse: () => void;
  onRun: () => void;
  onStop: () => void;
}

export function ParamsRail(props: Props) {
  const { defaults, params, onChange, audio, running, onBrowse, onRun, onStop } = props;
  const label = "block text-sm font-medium text-slate-600 dark:text-slate-300";
  const select =
    "mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100";
  return (
    <aside className="w-72 shrink-0 space-y-4 overflow-y-auto border-r border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div>
        <span className={label}>Audio file</span>
        <button
          onClick={onBrowse}
          className="mt-1 w-full rounded-lg border border-slate-300 px-2.5 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          {audio ? audio.name : "Browse…"}
        </button>
        {audio && (
          <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">
            {(audio.sizeBytes / (1024 * 1024)).toFixed(1)} MB
          </p>
        )}
      </div>

      <div>
        <label className={label}>Model</label>
        <select
          className={select}
          value={params.model}
          disabled={running}
          onChange={(e) => onChange({ ...params, model: e.target.value })}
        >
          {(defaults?.models ?? []).map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div>
        <label className={label}>Language</label>
        <select
          className={select}
          value={params.language}
          disabled={running}
          onChange={(e) => onChange({ ...params, language: e.target.value })}
        >
          {(defaults?.languages ?? []).map((l) => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>
      </div>

      <div>
        <label className={label}>Task</label>
        <div className="mt-1 flex gap-3 text-sm">
          {(["transcribe", "translate"] as const).map((t) => (
            <label key={t} className="flex items-center gap-1.5">
              <input
                type="radio"
                name="task"
                checked={params.task === t}
                disabled={running}
                onChange={() => onChange({ ...params, task: t })}
              />
              {t}
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className={label}>Beam size (0 = greedy)</label>
        <input
          type="number"
          min={0}
          className={select}
          value={params.beamSize}
          disabled={running}
          onChange={(e) => onChange({ ...params, beamSize: Number(e.target.value) })}
        />
      </div>

      <div>
        <label className={label}>Initial prompt</label>
        <input
          value={params.initialPrompt}
          disabled={running}
          onChange={(e) => onChange({ ...params, initialPrompt: e.target.value })}
          placeholder="Optional prompt for the first window"
          className={select}
        />
      </div>

      <div>
        <label className={label}>Device</label>
        <select
          className={select}
          value={params.device}
          disabled={running}
          onChange={(e) => onChange({ ...params, device: e.target.value })}
        >
          {(defaults?.devices ?? ["wgpu", "cpu"]).map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      <div className="flex gap-2 pt-1">
        <button
          onClick={onRun}
          disabled={running || !audio}
          className="flex-1 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Working…" : "Transcribe"}
        </button>
        <button
          onClick={onStop}
          disabled={!running}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          Stop
        </button>
      </div>
    </aside>
  );
}