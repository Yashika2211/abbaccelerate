import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { money, num, titleCase } from "../lib/format";
import { useSession } from "../lib/session";
import type { Impact, Leaderboard as Board } from "../lib/types";
import { Badge, Card, EmptyState, ErrorNote, Stat } from "../components/ui";

export default function Leaderboard() {
  const [session] = useSession();
  const [board, setBoard] = useState<Board | null>(null);
  const [impact, setImpact] = useState<Impact | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!session.runId) return;
    api.leaderboard(session.runId).then(setBoard).catch((e) => setError(String(e)));
    api.impact(session.runId).then(setImpact).catch(() => {});
  }, [session.runId]);

  if (!session.runId) {
    return <EmptyState title="No run selected" detail="Start a run to see a leaderboard." />;
  }
  if (error) {
    return (
      <Card title="Leaderboard">
        <ErrorNote error={error} />
      </Card>
    );
  }
  if (!board || board.empty_reason || board.rows.length === 0) {
    return (
      <EmptyState
        title="Nothing to rank yet"
        detail={board?.empty_reason ?? "This run has not reached the decision stage."}
      />
    );
  }

  const currency = board.currency ?? "INR";
  const kind = board.rows[0].operating_point_kind;
  const isThreshold = kind === "threshold";
  const metricKey = isThreshold ? "pr_auc" : "rmse";
  const metricLabel = isThreshold ? "PR-AUC" : "RMSE";

  // Which model a conventional AutoML leaderboard would have crowned.
  const metricChampion = [...board.rows].sort((a, b) => {
    const av = a.metrics[metricKey] ?? (isThreshold ? -1 : Infinity);
    const bv = b.metrics[metricKey] ?? (isThreshold ? -1 : Infinity);
    return isThreshold ? bv - av : av - bv;
  })[0];
  const costChampion = board.rows[0];
  const inverted = metricChampion.model_id !== costChampion.model_id;

  return (
    <div className="space-y-5">
      {inverted ? (
        <Card title="The best model on paper is not the best model here">
          <p className="text-sm leading-relaxed">
            <span className="font-medium">{metricChampion.model_name}</span> wins on {metricLabel}{" "}
            ({num(metricChampion.metrics[metricKey], 4)} against{" "}
            {num(costChampion.metrics[metricKey], 4)}). It also costs{" "}
            <span className="text-alarm font-medium">
              {money(
                metricChampion.cost_per_asset_year - costChampion.cost_per_asset_year,
                currency,
              )}
            </span>{" "}
            more per asset per year. Kairos ranks by cost, so it picks{" "}
            <span className="font-medium">{costChampion.model_name}</span>.
          </p>
        </Card>
      ) : (
        <Card title="Metric and economics agree here">
          <p className="text-muted text-sm">
            {costChampion.model_name} is both the best on {metricLabel} and the cheapest, so there
            is no trade-off to report. Kairos says so rather than inventing a disagreement.
          </p>
        </Card>
      )}

      <Card
        title="Leaderboard"
        subtitle={`Ranked by expected annual cost, not by ${metricLabel}. Computed in ${Math.round(board.elapsed_ms)}ms.`}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted border-edge border-b">
              <tr className="text-left">
                <th className="pb-2 font-medium">#</th>
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">
                  {isThreshold ? "Threshold" : "Lead time"}
                </th>
                <th className="pb-2 text-right font-medium">Cost / asset / year</th>
                <th className="pb-2 text-right font-medium">Cost regret</th>
                <th className="pb-2 text-right font-medium">{metricLabel}</th>
              </tr>
            </thead>
            <tbody>
              {board.rows.map((row, i) => (
                <tr
                  key={row.model_id}
                  className={`border-edge/60 border-b last:border-0 ${i === 0 ? "bg-signal/5" : ""}`}
                >
                  <td className="py-2 tabular-nums">{i + 1}</td>
                  <td className="py-2">
                    <span className="font-medium">{row.model_name}</span>
                    {i === 0 && (
                      <span className="ml-2">
                        <Badge tone="good">cheapest</Badge>
                      </span>
                    )}
                    {row.model_id === metricChampion.model_id && inverted && (
                      <span className="ml-2">
                        <Badge tone="warn">best {metricLabel}</Badge>
                      </span>
                    )}
                    {row.degenerate && (
                      <p className="text-muted mt-1 max-w-xl text-[11px] leading-relaxed">
                        {row.degenerate_reason}
                      </p>
                    )}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {isThreshold
                      ? row.operating_point.toFixed(3)
                      : `${row.operating_point.toFixed(0)} cyc`}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {money(row.cost_per_asset_year, currency)}
                  </td>
                  <td
                    className={`py-2 text-right tabular-nums ${row.cost_regret > 0 ? "text-alarm" : "text-muted"}`}
                  >
                    {row.cost_regret > 0 ? `+${money(row.cost_regret, currency)}` : "—"}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {num(row.metrics[metricKey], isThreshold ? 4 : 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {board.baselines && (
        <Card
          title="Honest baselines"
          subtitle="Computed on the same test set. Fixed-interval has its interval optimised too."
        >
          <div className="space-y-2">
            {board.baselines.baselines.map((b) => {
              const max = Math.max(...board.baselines!.baselines.map((x) => x.cost_per_asset_year));
              const width = max > 0 ? (b.cost_per_asset_year / max) * 100 : 0;
              const isKairos = b.key === "kairos";
              return (
                <div key={b.key} className="flex items-center gap-3">
                  <span className="w-56 shrink-0 text-xs">{b.name}</span>
                  <div className="bg-edge/40 h-5 flex-1 overflow-hidden rounded">
                    <div
                      className={`h-full ${isKairos ? "bg-signal" : "bg-muted/50"}`}
                      style={{ width: `${width}%` }}
                    />
                  </div>
                  <span className="w-24 shrink-0 text-right text-xs tabular-nums">
                    {money(b.cost_per_asset_year, currency)}
                  </span>
                </div>
              );
            })}
          </div>
          <p
            className={`mt-4 text-sm leading-relaxed ${board.baselines.kairos_wins ? "" : "text-alarm"}`}
          >
            {board.baselines.verdict}
          </p>
        </Card>
      )}

      {impact && !impact.empty_reason && (
        <Card title="Impact" subtitle={impact.annualisation_note}>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Champion" value={impact.champion} />
            <Stat
              label={titleCase(impact.operating_point_kind ?? "operating point")}
              value={
                impact.operating_point_kind === "threshold"
                  ? impact.operating_point.toFixed(3)
                  : `${impact.operating_point.toFixed(0)} cyc`
              }
            />
            <Stat
              label="Cost / asset / year"
              value={money(impact.cost_per_asset_year, currency)}
            />
            <Stat
              label="Saved vs next best"
              value={money(impact.savings_per_asset_year ?? 0, currency)}
              tone={impact.kairos_wins ? "good" : "bad"}
            />
          </dl>
          <p className="text-muted mt-3 text-[11px]">
            Narration: {impact.llm_used ? "written by Claude from computed values" : "deterministic templates (no API key set)"}.
            Either way the numbers come from the cost engine, never from a language model.
          </p>
        </Card>
      )}
    </div>
  );
}
