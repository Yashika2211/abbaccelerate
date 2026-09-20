import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AGENT_EVENTS, api, subscribe } from "../lib/api";
import { duration, num, titleCase } from "../lib/format";
import { useSession } from "../lib/session";
import type { AgentEvent, RunDetail, TimelineEvent } from "../lib/types";
import { Badge, Button, Card, EmptyState, ErrorNote } from "../components/ui";

const NODE_LABELS: Record<string, string> = {
  profile_dataset: "Profile dataset",
  diagnose: "Diagnose and plan",
  human_checkpoint: "Your approval",
  engineer_features: "Engineer features",
  train_candidates: "Train candidates",
  evaluate: "Evaluate + quality gate",
  decide: "Decision layer",
  explain: "Explain (SHAP)",
  schedule: "Schedule work orders",
  narrate: "Write work orders",
  report: "Executive summary",
  recover: "Recover",
};

type Trial = { model_name: string; ok: boolean; fit_seconds: number; metrics: Record<string, number> };

export default function Run() {
  const [session] = useSession();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);
  const navigate = useNavigate();
  const bottom = useRef<HTMLDivElement>(null);

  const runId = session.runId;

  const refresh = useCallback(async () => {
    if (!runId) return;
    try {
      const [detail, events] = await Promise.all([api.run(runId), api.timeline(runId)]);
      setRun(detail);
      setTimeline(events.filter((e) => e.status !== "started"));
    } catch (e) {
      setError(String(e));
    }
  }, [runId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Live stream. Every event triggers a refresh of the persisted timeline rather
  // than being rendered directly, so the browser and the database cannot diverge.
  useEffect(() => {
    if (!runId) return;
    return subscribe(
      `/api/runs/${runId}/stream`,
      (name, data) => {
        const event = data as AgentEvent;
        if (name === "trial") {
          setTrials((prev) => [
            ...prev,
            {
              model_name: event.model_name ?? "?",
              ok: Boolean(event.ok),
              fit_seconds: event.fit_seconds ?? 0,
              metrics: event.metrics ?? {},
            },
          ]);
        }
        if (name === "error") setError(event.message ?? "run failed");
        refresh();
      },
      AGENT_EVENTS,
    );
  }, [runId, refresh]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [timeline.length]);

  async function decide(approved: boolean) {
    if (!runId) return;
    setApproving(true);
    setError(null);
    try {
      await api.approve(runId, approved);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setApproving(false);
    }
  }

  if (!runId) {
    return (
      <EmptyState
        title="No run in progress"
        detail="Set your plant economics in Cost Studio and start a run. The agent will stop and ask before it trains anything."
      />
    );
  }

  const plan = (run?.plan ?? null) as Record<string, unknown> | null;
  const awaiting = run?.awaiting_approval;

  return (
    <div className="space-y-5">
      <Card
        title="Agent run"
        subtitle={`${runId.slice(0, 8)} · ${run?.task_type?.replace(/_/g, " ") ?? "…"}`}
        right={
          <div className="flex items-center gap-2">
            <Badge
              tone={
                run?.status === "complete"
                  ? "good"
                  : run?.status === "failed"
                    ? "bad"
                    : run?.status === "awaiting_approval"
                      ? "warn"
                      : "accent"
              }
            >
              {run?.status ?? "…"}
            </Badge>
            {run?.status === "complete" && (
              <Button onClick={() => navigate("/leaderboard")}>See leaderboard →</Button>
            )}
          </div>
        }
      >
        <p className="text-muted text-xs">
          Every step below is written to an audit table as it happens, with its own duration and
          reasoning. This panel is that table, streamed live.
        </p>
        <ErrorNote error={error ?? run?.error ?? null} />
      </Card>

      {awaiting && plan && (
        <Card
          title="The agent is waiting for you"
          subtitle="Execution has genuinely stopped. State is checkpointed; approving resumes it."
        >
          <dl className="grid gap-3 text-xs sm:grid-cols-2">
            <div>
              <dt className="text-muted">Task</dt>
              <dd className="font-medium">{String(plan.task_type ?? "—").replace(/_/g, " ")}</dd>
            </div>
            <div>
              <dt className="text-muted">Target</dt>
              <dd className="font-mono">{String(plan.target ?? "—")}</dd>
            </div>
            <div>
              <dt className="text-muted">Feature strategy</dt>
              <dd>
                {String(plan.feature_strategy ?? "—").replace(/_/g, " ")}
                {Array.isArray(plan.window_sizes) && plan.window_sizes.length > 0 && (
                  <span className="text-muted"> · windows {(plan.window_sizes as number[]).join(", ")}</span>
                )}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Candidates</dt>
              <dd>{(plan.candidate_models as string[] | undefined)?.join(", ") ?? "—"}</dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-muted">Columns it will drop</dt>
              <dd className="mt-1 flex flex-wrap gap-1">
                {((plan.drop_columns as string[] | undefined) ?? []).map((c) => (
                  <span key={c} className="bg-edge rounded px-1.5 py-0.5 font-mono text-[11px]">
                    {c}
                  </span>
                ))}
              </dd>
            </div>
            {typeof plan.reasoning === "string" && (
              <div className="sm:col-span-2">
                <dt className="text-muted">Reasoning</dt>
                <dd className="mt-1 leading-relaxed">{plan.reasoning}</dd>
              </div>
            )}
          </dl>

          <div className="mt-4 flex gap-2">
            <Button onClick={() => decide(true)} disabled={approving}>
              {approving ? "resuming…" : "Approve and train"}
            </Button>
            <Button variant="danger" onClick={() => decide(false)} disabled={approving}>
              Reject
            </Button>
          </div>
        </Card>
      )}

      <Card title="Timeline" subtitle="The audit trail, as it is written.">
        {timeline.length === 0 ? (
          <p className="text-muted text-xs">waiting for the first step…</p>
        ) : (
          <ol className="space-y-0">
            {timeline.map((e) => (
              <li key={e.sequence} className="border-edge/60 flex gap-3 border-b py-2.5 last:border-0">
                <span
                  className={`mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full ${
                    e.status === "failed" ? "bg-alarm" : "bg-signal"
                  }`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="text-sm font-medium">
                      {NODE_LABELS[e.node] ?? titleCase(e.node)}
                    </span>
                    {e.duration_ms !== null && (
                      <span className="text-muted text-[11px] tabular-nums">
                        {duration(e.duration_ms)}
                      </span>
                    )}
                    {e.status === "failed" && <Badge tone="bad">failed</Badge>}
                  </div>
                  {e.summary && <p className="mt-0.5 text-xs">{e.summary}</p>}
                  {e.reasoning && (
                    <p className="text-muted mt-0.5 text-[11px] leading-relaxed">{e.reasoning}</p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
        <div ref={bottom} />
      </Card>

      {trials.length > 0 && (
        <Card title="Candidates" subtitle="Streamed as each one finishes training.">
          <table className="w-full text-xs">
            <thead className="text-muted border-edge border-b">
              <tr className="text-left">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">Fit</th>
                <th className="pb-2 text-right font-medium">PR-AUC</th>
                <th className="pb-2 text-right font-medium">RMSE</th>
              </tr>
            </thead>
            <tbody>
              {trials.map((t, i) => (
                <tr key={i} className="border-edge/60 border-b last:border-0">
                  <td className="py-1.5">
                    {t.model_name} {!t.ok && <Badge tone="bad">failed</Badge>}
                  </td>
                  <td className="py-1.5 text-right tabular-nums">{t.fit_seconds.toFixed(1)}s</td>
                  <td className="py-1.5 text-right tabular-nums">
                    {t.metrics.pr_auc !== undefined ? num(t.metrics.pr_auc, 4) : "—"}
                  </td>
                  <td className="py-1.5 text-right tabular-nums">
                    {t.metrics.rmse !== undefined ? num(t.metrics.rmse, 3) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {run && run.findings.length > 0 && (
        <Card title="Findings" subtitle="What the agent noticed, in the order it noticed it.">
          <ul className="space-y-1.5 text-xs leading-relaxed">
            {run.findings.map((f, i) => (
              <li key={i} className={f.startsWith("TARGET LEAKAGE") ? "text-alarm" : ""}>
                – {f}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
