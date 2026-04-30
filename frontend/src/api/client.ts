const BASE = import.meta.env.VITE_API_URL ?? "";

export interface InferResult {
  id: string;
  filename?: string;
  created_at?: string;
  labels: string[];
  threshold: number;
  original_height: number;
  original_width: number;
  areas: Record<string, number>;
  inference_time_s: number;
}

export interface ModelInfo {
  labels: string[];
  threshold: number;
  device: string;
  config_path: string;
  config_exists: boolean;
  checkpoint_name: string;
  checkpoint_exists: boolean;
}

export interface HealthInfo {
  status: string;
  model_loaded: boolean;
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${BASE}/model`);
  if (!res.ok) throw new Error(`Model info failed: ${res.status}`);
  return res.json();
}

export async function runInference(
  file: File,
  threshold?: number
): Promise<InferResult> {
  const form = new FormData();
  form.append("file", file);
  if (threshold !== undefined) {
    form.append("threshold", String(threshold));
  }
  const res = await fetch(`${BASE}/infer`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? res.statusText);
  }
  return res.json();
}

export async function fetchResults(): Promise<InferResult[]> {
  const res = await fetch(`${BASE}/results`);
  if (!res.ok) throw new Error(`Result history failed: ${res.status}`);
  const data = await res.json();
  return data.results ?? [];
}

export async function deleteResult(id: string): Promise<void> {
  const res = await fetch(`${BASE}/results/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? res.statusText);
  }
}

export function resultImageUrl(id: string): string {
  return `${BASE}/results/${id}/image`;
}

export function resultOverlayUrl(id: string): string {
  return `${BASE}/results/${id}/overlay`;
}

export function resultLabelOverlayUrl(id: string, label: string): string {
  return `${BASE}/results/${id}/overlay/${label}`;
}

export function resultMaskUrl(id: string): string {
  return `${BASE}/results/${id}/mask`;
}

export function resultLabelMaskUrl(id: string, label: string): string {
  return `${BASE}/results/${id}/mask/${label}`;
}
