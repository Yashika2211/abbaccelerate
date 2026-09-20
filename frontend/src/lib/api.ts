/** Thin API client. Every call fails loudly so pages can render an empty state. */

import type {
  BuiltinDataset,
  CostConfig,
  DatasetDetail,
  Health,
  Impact,
  Leaderboard,
  RunDetail,
  TimelineEvent,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* the status line is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const getJSON = <T,>(path: string) => request<T>(path);

export const postJSON = <T,>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  health: () => getJSON<Health>("/api/health"),
  builtins: () => getJSON<BuiltinDataset[]>("/api/datasets/builtin"),
  createDataset: (builtin: string) =>
    postJSON<{ dataset_id: string; name: string; rows: number }>("/api/datasets", { builtin }),
  uploadDataset: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/datasets/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
    return (await res.json()) as { dataset_id: string; name: string; rows: number };
  },
  datasetProfile: (id: string) => getJSON<DatasetDetail>(`/api/datasets/${id}/profile`),
  createRun: (dataset_id: string, cost_config: CostConfig | null, time_budget_s: number) =>
    postJSON<{ run_id: string; status: string; checkpointer: string; stream_url: string }>(
      "/api/runs",
      { dataset_id, cost_config, time_budget_s },
    ),
  run: (id: string) => getJSON<RunDetail>(`/api/runs/${id}`),
  timeline: (id: string) => getJSON<TimelineEvent[]>(`/api/runs/${id}/timeline`),
  approve: (id: string, approved: boolean, plan?: Record<string, unknown> | null) =>
    postJSON<{ resumed: boolean }>(`/api/runs/${id}/approve`, { approved, plan }),
  leaderboard: (id: string) => getJSON<Leaderboard>(`/api/runs/${id}/leaderboard`),
  impact: (id: string) => getJSON<Impact>(`/api/runs/${id}/impact`),
  workorders: (id: string) => getJSON<Record<string, unknown>>(`/api/runs/${id}/workorders`),
};

/**
 * Subscribe to a Server-Sent Events endpoint.
 * Returns an unsubscribe function. `names` lists the event types to listen for —
 * EventSource requires each to be registered explicitly.
 */
export function subscribe(
  path: string,
  onEvent: (name: string, data: Record<string, unknown>) => void,
  names: string[],
): () => void {
  const source = new EventSource(path);
  const handler = (name: string) => (ev: MessageEvent) => {
    try {
      onEvent(name, JSON.parse(ev.data));
    } catch {
      onEvent(name, { raw: ev.data });
    }
  };
  for (const name of names) source.addEventListener(name, handler(name));
  source.onmessage = handler("message");
  return () => source.close();
}

export const AGENT_EVENTS = [
  "run_started",
  "node",
  "trial",
  "awaiting_approval",
  "run_status",
  "complete",
  "error",
  "stream_end",
];

export const charts = {
  costCurve: (id: string) => getJSON<Record<string, unknown>>(`/api/runs/${id}/cost-curve`),
  calibration: (id: string) => getJSON<Record<string, unknown>>(`/api/runs/${id}/calibration`),
  explain: (id: string) => getJSON<Record<string, unknown>>(`/api/runs/${id}/explain`),
  degradation: (id: string) => getJSON<Record<string, unknown>>(`/api/runs/${id}/degradation`),
};
