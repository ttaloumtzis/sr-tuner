import type { DatasetInfo, JobAccepted, JobStatus, WorkspaceInfo, TrainParams, InferParams, EnvInfo, DatasetBuildParams, DatasetValidateParams, DatasetHealthParams, DatasetMergeParams, ExportParams, ModelInstance, ModelVersion } from "./api-types";

let BASE_URL = "http://127.0.0.1:8765";

export async function initApiUrl(): Promise<void> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    BASE_URL = await invoke<string>("get_server_url");
  } catch {
    // Running in browser dev mode — Vite proxy handles /api/*
  }
}

export function getBaseUrl(): string {
  return BASE_URL;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${method} ${path} ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Workspace ───────────────────────────────────────────────────────────

export const getWorkspace = () => request<WorkspaceInfo>("GET", "/api/workspace");
export const initWorkspace = (path?: string) => request<WorkspaceInfo>("POST", "/api/workspace/init", { path });

// ── Models ──────────────────────────────────────────────────────────────

export const listModels = () => request<{ name: string }[]>("GET", "/api/models");
export const listInstances = () => request<ModelInstance[]>("GET", "/api/models/instances");
export const getInstance = (name: string) => request<ModelInstance>("GET", `/api/models/instances/${name}`);
export const exportModel = (name: string, params: ExportParams) => request<{ output: string }>("POST", `/api/models/instances/${name}/export`, params);
export const getInstanceVersions = (name: string) => request<ModelVersion[]>("GET", `/api/models/instances/${encodeURIComponent(name)}/versions`);
export const deleteInstance = (name: string) => request<{ deleted: string }>("DELETE", `/api/models/instances/${encodeURIComponent(name)}`);
export const createInstance = (name: string, architecture: string, config: Record<string, unknown>) =>
  request<ModelInstance>("POST", "/api/models/instances", { name, architecture, config });

// ── Training ────────────────────────────────────────────────────────────

export const startTraining = (params: TrainParams) => request<JobAccepted>("POST", "/api/train/start", params);
export const validateDataset = (params: { dataset: string }) => request<{ valid: boolean; problems: string[] }>("POST", "/api/train/validate-dataset", params);
export const listDatasets = (scale?: number) => {
  const qs = scale !== undefined ? `?scale=${scale}` : "";
  return request<DatasetInfo[]>("GET", `/api/datasets${qs}`);
};

// ── Inference ───────────────────────────────────────────────────────────

export const startInference = (params: InferParams) => request<JobAccepted>("POST", "/api/infer/start", params);

// ── Datasets ────────────────────────────────────────────────────────────

export const buildDataset = (params: DatasetBuildParams) => request<JobAccepted>("POST", "/api/datasets/build", params);
export const validateDatasetPath = (params: DatasetValidateParams) => request<{ valid: boolean; problems: string[]; num_pairs: number }>("POST", "/api/datasets/validate", params);
export const startValidateDataset = (params: DatasetValidateParams) => request<JobAccepted>("POST", "/api/datasets/validate-async", params);
export const healthCheck = (params: DatasetHealthParams) => request<JobAccepted>("POST", "/api/datasets/health", params);
export const getDatasetHealth = (path: string) => request<Record<string, unknown> | null>("GET", `/api/datasets/health?path=${encodeURIComponent(path)}`);
export const mergeDatasets = (params: DatasetMergeParams) => request<JobAccepted>("POST", "/api/datasets/merge", params);
export const pruneBlackFrames = (params: { path: string; black_frames: string[] }) => request<JobAccepted>("POST", "/api/datasets/prune", params);

// ── Jobs ────────────────────────────────────────────────────────────────

export const listJobs = () => request<{ jobs: JobStatus[] }>("GET", "/api/jobs");
export const getJobStatus = (jobId: string) => request<JobStatus>("GET", `/api/jobs/${jobId}`);
export const cancelJob = (jobId: string) => request<{ status: string }>("POST", `/api/jobs/${jobId}/cancel`);

// ── Env ─────────────────────────────────────────────────────────────────

export const getEnv = () => request<EnvInfo>("GET", "/api/env");