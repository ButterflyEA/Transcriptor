export interface Defaults {
  models: string[];
  languages: string[];
  defaultModel: string;
  defaultTask: string;
  defaultBeamSize: number;
  devices: string[];
}

export interface TaskParams {
  model: string;
  language: string;
  task: "transcribe" | "translate";
  beamSize: number;
  initialPrompt: string;
  device: string;
}

export interface ProgressDto {
  kind: string;
  message: string;
  fraction: number | null;
}

export interface SegmentDto {
  start: number;
  end: number;
  text: string;
}

export interface DoneDto {
  status: "finished" | "cancelled" | "error";
  segments: SegmentDto[];
  message: string | null;
}

export interface AudioInfo {
  path: string;
  name: string;
  sizeBytes: number;
}

export function clock(ms: number): string {
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(milli).padStart(3, "0")}`;
}