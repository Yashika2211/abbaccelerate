/** Thin API client. Every call fails loudly here so pages can render empty states. */

export type Health = {
  status: "ok" | "degraded";
  version: string;
  env: string;
  uptime_s: number;
  components: { postgres: string; mlflow: string; llm: string };
};

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return (await res.json()) as T;
}

export async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return (await res.json()) as T;
}

/**
 * Subscribe to a Server-Sent Events endpoint.
 * `onEvent` receives (eventName, parsedData). Returns an unsubscribe function.
 */
export function subscribe(
  path: string,
  onEvent: (name: string, data: Record<string, unknown>) => void,
  names: string[] = ["message"],
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
