import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { pct } from "../lib/format";
import { useSession } from "../lib/session";
import type { BuiltinDataset, DatasetDetail } from "../lib/types";
import { Badge, Button, Card, EmptyState, ErrorNote } from "../components/ui";

export default function Data() {
  const [builtins, setBuiltins] = useState<BuiltinDataset[]>([]);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [session, update] = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    api.builtins().then(setBuiltins).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!session.datasetId) return;
    api.datasetProfile(session.datasetId).then(setDetail).catch((e) => setError(String(e)));
  }, [session.datasetId]);

  async function select(key: string) {
    setBusy(key);
    setError(null);
    try {
      const created = await api.createDataset(key);
      update({ datasetId: created.dataset_id, datasetName: created.name, runId: null });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function upload(file: File) {
    setBusy("upload");
    setError(null);
    try {
      const created = await api.uploadDataset(file);
      update({ datasetId: created.dataset_id, datasetName: created.name, runId: null });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  const profile = detail?.profile ?? null;

  return (
    <div className="space-y-5">
      <Card title="Choose a dataset" subtitle="Two real, citable public datasets, or your own CSV.">
        <div className="grid gap-3 sm:grid-cols-2">
          {builtins.map((b) => (
            <button
              key={b.key}
              onClick={() => select(b.key)}
              disabled={busy !== null}
              className={`border-edge hover:border-accent/60 rounded border px-4 py-3 text-left transition disabled:opacity-50 ${
                detail?.name?.includes(b.name.split(" ")[0]) ? "border-accent/60" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{b.name}</span>
                {busy === b.key && <span className="text-muted text-xs">loading…</span>}
              </div>
              <p className="text-muted mt-1 text-xs">{b.description}</p>
            </button>
          ))}
        </div>

        <label className="border-edge mt-3 flex cursor-pointer items-center justify-between rounded border border-dashed px-4 py-3 text-xs">
          <span className="text-muted">
            …or upload a CSV. The profiler will propose a task and target.
          </span>
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
          <span className="text-accent">{busy === "upload" ? "uploading…" : "Browse"}</span>
        </label>

        <ErrorNote error={error} />
      </Card>

      {!profile && !error && (
        <EmptyState
          title="No dataset loaded yet"
          detail="Pick one above. Kairos profiles it before it asks you anything about the task."
        />
      )}

      {profile && (
        <>
          <Card
            title={`Profile — ${profile.name}`}
            subtitle={detail?.citation}
            right={
              <Button onClick={() => navigate("/cost")}>Set plant economics →</Button>
            }
          >
            <dl className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div>
                <dt className="text-muted">Rows</dt>
                <dd className="text-base font-semibold tabular-nums">
                  {profile.n_rows.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-muted">Columns</dt>
                <dd className="text-base font-semibold tabular-nums">{profile.n_columns}</dd>
              </div>
              <div>
                <dt className="text-muted">Proposed task</dt>
                <dd className="text-base font-semibold">
                  {profile.proposed_task_type?.replace(/_/g, " ") ?? "—"}
                </dd>
              </div>
              <div>
                <dt className="text-muted">Target</dt>
                <dd className="text-base font-semibold">{profile.proposed_target ?? "—"}</dd>
              </div>
            </dl>

            <div className="mt-4 flex flex-wrap gap-2">
              <Badge tone={profile.constant_columns.length ? "warn" : "neutral"}>
                {profile.constant_columns.length} dead columns
              </Badge>
              <Badge tone={profile.leakage_warnings.length ? "bad" : "good"}>
                {profile.leakage_warnings.length} leakage convictions
              </Badge>
              <Badge tone="neutral">{profile.recommended_drops.length} to drop</Badge>
              {profile.class_balance && (
                <Badge tone="accent">
                  {pct(profile.class_balance["1"] ?? 0, 2)} positive
                </Badge>
              )}
              <Badge tone="neutral">source: {detail?.source}</Badge>
            </div>
          </Card>

          <Card title="What the profiler found" subtitle="Computed before any model was trained.">
            <ul className="space-y-2 text-xs leading-relaxed">
              {profile.findings.map((finding, i) => {
                const isLeak = finding.startsWith("TARGET LEAKAGE");
                return (
                  <li
                    key={i}
                    className={`flex gap-2 ${isLeak ? "text-alarm" : "text-slate-200"}`}
                  >
                    <span className="text-muted shrink-0">{isLeak ? "!" : "–"}</span>
                    <span>{finding}</span>
                  </li>
                );
              })}
            </ul>
          </Card>

          {profile.leakage_warnings.length > 0 && (
            <Card
              title="Target leakage"
              subtitle="Convicted on evidence: P(target | column fires) against the base rate."
            >
              <table className="w-full text-xs">
                <thead className="text-muted border-edge border-b">
                  <tr className="text-left">
                    <th className="pb-2 font-medium">Column</th>
                    <th className="pb-2 font-medium">Lift</th>
                    <th className="pb-2 font-medium">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.leakage_warnings.map((w) => (
                    <tr key={w.column} className="border-edge/60 border-b last:border-0">
                      <td className="py-2 font-mono">{w.column}</td>
                      <td className="text-alarm py-2 tabular-nums">{w.score.toFixed(1)}×</td>
                      <td className="text-muted py-2">{w.evidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {profile.documented_leakage_undetected.length > 0 && (
                <p className="text-muted mt-3 text-[11px]">
                  Dropped on provenance rather than evidence:{" "}
                  <span className="font-mono">
                    {profile.documented_leakage_undetected.join(", ")}
                  </span>
                  . Documented as failure-mode flags, but no statistical test convicts them.
                </p>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
