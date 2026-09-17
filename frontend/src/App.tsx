import { useEffect, useState } from "react";
import { getJSON, subscribe, type Health } from "./lib/api";

function Dot({ up }: { up: boolean }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${up ? "bg-signal" : "bg-alarm"}`}
      aria-label={up ? "up" : "down"}
    />
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      getJSON<Health>("/api/health")
        .then((h) => alive && (setHealth(h), setError(null)))
        .catch((e) => alive && setError(String(e)));
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(
    () => subscribe("/api/events/demo", (_n, d) => setTick(Number(d.n)), ["tick"]),
    [],
  );

  return (
    <main className="mx-auto flex min-h-full max-w-2xl flex-col justify-center gap-6 p-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Kairos</h1>
        <p className="text-muted text-sm">
          Every other tool stops at a probability. Kairos stops at a decision, in rupees.
        </p>
      </header>

      <section className="border-edge bg-panel rounded-lg border p-5">
        <h2 className="mb-3 text-xs font-medium tracking-widest text-muted uppercase">
          Phase 0 — system check
        </h2>

        {error && !health && <p className="text-alarm text-sm">API unreachable — {error}</p>}

        {health && (
          <dl className="grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-muted">API</dt>
            <dd className="flex items-center gap-2">
              <Dot up={health.status === "ok"} /> {health.status} · v{health.version}
            </dd>
            <dt className="text-muted">Postgres</dt>
            <dd className="flex items-center gap-2">
              <Dot up={health.components.postgres === "up"} /> {health.components.postgres}
            </dd>
            <dt className="text-muted">MLflow</dt>
            <dd className="flex items-center gap-2">
              <Dot up={health.components.mlflow === "up"} /> {health.components.mlflow}
            </dd>
            <dt className="text-muted">LLM</dt>
            <dd className="flex items-center gap-2">
              <Dot up={health.components.llm === "configured"} /> {health.components.llm}
            </dd>
            <dt className="text-muted">SSE</dt>
            <dd className="flex items-center gap-2">
              <Dot up={tick !== null} />
              {tick === null ? "waiting for first event" : `tick ${tick}`}
            </dd>
          </dl>
        )}

        {!health && !error && <p className="text-muted text-sm">checking…</p>}
      </section>

      <p className="text-muted text-xs">
        Values on this page are read from <code>/api/health</code> and the live event stream.
        Nothing here is mocked.
      </p>
    </main>
  );
}
