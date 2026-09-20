import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { money } from "../../lib/format";
import { AXIS_PROPS, GRID_PROPS, INK, SERIES, STATUS } from "../../lib/viz";
import { ChartFrame, LegendKey, TooltipCard } from "./ChartFrame";

export type CostCurvePoint = {
  x: number;
  cost: number;
  cost_per_asset_year: number;
  tp?: number;
  fp?: number;
  fn?: number;
  caught?: number;
  missed?: number;
  wasted_life_cycles?: number;
};

export type CostCurveData = {
  kind: "threshold" | "lead_time";
  model_name: string;
  optimum: number;
  bayes_threshold?: number;
  currency: string;
  chosen_on: string;
  degenerate?: boolean;
  degenerate_reason?: string | null;
  calibration_warning?: string | null;
  points: CostCurvePoint[];
  empty_reason?: string;
};

/**
 * Expected cost against the decision variable, with the chosen operating point
 * marked. One series, so no legend box — the title names it and the optimum is
 * direct-labelled. Reference lines are drawn in ink, not series colour, because
 * they are annotations rather than data.
 */
export default function CostCurve({ data }: { data: CostCurveData }) {
  const isThreshold = data.kind === "threshold";
  const optimal = data.points.reduce(
    (best, p) => (p.cost < best.cost ? p : best),
    data.points[0],
  );

  // A dense sweep (1,100+ points on a classification run) is more marks than the
  // pixel grid can hold. Thin it evenly, always keeping the optimum.
  const MAX_POINTS = 240;
  const stride = Math.max(1, Math.floor(data.points.length / MAX_POINTS));
  const plotted = data.points.filter(
    (p, i) => i % stride === 0 || p.x === optimal.x || i === data.points.length - 1,
  );

  const label = isThreshold ? "Decision threshold" : "Lead time (cycles)";

  // With a 43:1 consequence ratio the optimum lands near 0.01, and everything
  // interesting happens in the first 3% of the axis — over a [0,1] domain the
  // hero chart reads as a flat line with a spike. When the minimum sits in the
  // left tenth, a detail panel is drawn beside the full sweep rather than
  // clipping the axis, so the reader sees both the shape and the choice.
  const domainMax = plotted[plotted.length - 1]?.x ?? 1;
  const needsDetail = isThreshold && optimal.x < domainMax * 0.1;
  const detailMax = Math.min(domainMax, Math.max(optimal.x * 6, domainMax * 0.04));
  const detail = needsDetail ? plotted.filter((p) => p.x <= detailMax) : [];

  const axis = (
    <>
      <CartesianGrid {...GRID_PROPS} />
      <Tooltip
        cursor={{ stroke: INK.muted, strokeWidth: 1, strokeDasharray: "3 3" }}
        content={({ active, payload }) => {
          if (!active || !payload?.length) return null;
          const p = payload[0].payload as CostCurvePoint;
          const rows: [string, string][] = [
            [label, isThreshold ? p.x.toFixed(4) : String(p.x)],
            ["Cost / asset / yr", money(p.cost_per_asset_year, data.currency)],
          ];
          if (p.fn !== undefined) {
            rows.push(["Missed failures", String(p.fn)], ["False alarms", String(p.fp ?? 0)]);
          }
          if (p.missed !== undefined) {
            rows.push(
              ["Caught in time", String(p.caught ?? 0)],
              ["Missed", String(p.missed)],
              ["Life scrapped", `${Math.round(p.wasted_life_cycles ?? 0)} cyc`],
            );
          }
          return <TooltipCard rows={rows} />;
        }}
      />
    </>
  );

  return (
    <ChartFrame
      title={
        isThreshold
          ? "Expected cost against the decision threshold"
          : "Expected cost against intervention lead time"
      }
      subtitle={
        isThreshold
          ? `Swept on the ${data.chosen_on} set for ${data.model_name}. The minimum is the operating point Kairos ships — not 0.5, and not max-F1.`
          : `Swept on the ${data.chosen_on} set for ${data.model_name}. Alert too late and you pay for unplanned failures; too early and you scrap good component life. The bottom of that U is the recommendation.`
      }
      legend={
        <>
          <LegendKey colour={SERIES[0]} label="expected cost" />
          <LegendKey colour={STATUS.good} label={`chosen ${isThreshold ? "threshold" : "lead time"}`} dashed />
          {isThreshold && data.bayes_threshold !== undefined && (
            <LegendKey colour={INK.muted} label="closed-form Bayes optimum" dashed />
          )}
        </>
      }
      footnote={
        isThreshold && data.bayes_threshold !== undefined ? (
          <>
            The swept optimum is <strong>{optimal.x.toFixed(4)}</strong>; the closed-form Bayes
            threshold for these costs is <strong>{data.bayes_threshold.toFixed(4)}</strong>. On
            well-calibrated probabilities those agree — the gap is a calibration check, not a
            disagreement about the economics.
          </>
        ) : (
          <>
            Minimum at <strong>{optimal.x}</strong> cycles, costing{" "}
            <strong>{money(optimal.cost_per_asset_year, data.currency)}</strong> per asset per year.
          </>
        )
      }
      table={
        <table className="w-full text-[11px]">
          <thead className="text-muted sticky top-0 bg-[#111820]">
            <tr className="text-left">
              <th className="py-1 pr-4 font-medium">{label}</th>
              <th className="py-1 pr-4 text-right font-medium">Cost / asset / year</th>
              <th className="py-1 text-right font-medium">
                {isThreshold ? "False negatives" : "Missed"}
              </th>
            </tr>
          </thead>
          <tbody>
            {plotted.map((p) => (
              <tr key={p.x} className={p.x === optimal.x ? "text-signal" : ""}>
                <td className="py-0.5 pr-4 tabular-nums">
                  {isThreshold ? p.x.toFixed(4) : p.x}
                </td>
                <td className="py-0.5 pr-4 text-right tabular-nums">
                  {money(p.cost_per_asset_year, data.currency)}
                </td>
                <td className="py-0.5 text-right tabular-nums">{p.fn ?? p.missed ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
      height={300}
    >
      <div className={needsDetail ? "grid h-full gap-4 lg:grid-cols-[1.1fr_1fr]" : "h-full"}>
        <div className="h-full min-h-0">
          {needsDetail && (
            <p className="text-muted mb-1 text-[11px]">Full sweep, 0 to 1</p>
          )}
          <ResponsiveContainer width="100%" height={needsDetail ? "92%" : "100%"}>
            <LineChart data={plotted} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
              {axis}
              <XAxis
                dataKey="x"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => (isThreshold ? v.toFixed(2) : String(v))}
                label={{
                  value: label,
                  position: "insideBottom",
                  offset: -12,
                  fill: INK.muted,
                  fontSize: 11,
                }}
                {...AXIS_PROPS}
              />
              <YAxis
                tickFormatter={(v: number) => money(v, data.currency)}
                width={78}
                {...AXIS_PROPS}
              />
              <ReferenceLine
                x={data.optimum}
                stroke={STATUS.good}
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={
                  needsDetail
                    ? undefined
                    : {
                        value: isThreshold ? data.optimum.toFixed(3) : `${data.optimum} cyc`,
                        position: "top",
                        fill: STATUS.good,
                        fontSize: 11,
                      }
                }
              />
              {isThreshold && data.bayes_threshold !== undefined && (
                <ReferenceLine
                  x={data.bayes_threshold}
                  stroke={INK.muted}
                  strokeDasharray="2 4"
                  strokeWidth={1}
                />
              )}
              <Line
                type="monotone"
                dataKey="cost_per_asset_year"
                stroke={SERIES[0]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: SERIES[0], stroke: INK.surface, strokeWidth: 2 }}
                isAnimationActive={false}
              />
              <ReferenceDot
                x={optimal.x}
                y={optimal.cost_per_asset_year}
                r={5}
                fill={STATUS.good}
                stroke={INK.surface}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {needsDetail && (
          <div className="h-full min-h-0">
            <p className="text-muted mb-1 text-[11px]">
              Detail: 0 to {detailMax.toFixed(3)} — where the decision is actually made
            </p>
            <ResponsiveContainer width="100%" height="92%">
              <LineChart data={detail} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                {axis}
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={[0, detailMax]}
                  tickFormatter={(v: number) => v.toFixed(3)}
                  label={{
                    value: label,
                    position: "insideBottom",
                    offset: -12,
                    fill: INK.muted,
                    fontSize: 11,
                  }}
                  {...AXIS_PROPS}
                />
                <YAxis
                  tickFormatter={(v: number) => money(v, data.currency)}
                  width={78}
                  domain={["auto", "auto"]}
                  {...AXIS_PROPS}
                />
                <ReferenceLine
                  x={data.optimum}
                  stroke={STATUS.good}
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{
                    value: data.optimum.toFixed(3),
                    position: "top",
                    fill: STATUS.good,
                    fontSize: 11,
                  }}
                />
                {data.bayes_threshold !== undefined && (
                  <ReferenceLine
                    x={data.bayes_threshold}
                    stroke={INK.muted}
                    strokeDasharray="2 4"
                    strokeWidth={1}
                    label={{
                      value: "Bayes",
                      position: "insideTopRight",
                      fill: INK.muted,
                      fontSize: 10,
                    }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="cost_per_asset_year"
                  stroke={SERIES[0]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: SERIES[0], stroke: INK.surface, strokeWidth: 2 }}
                  isAnimationActive={false}
                />
                <ReferenceDot
                  x={optimal.x}
                  y={optimal.cost_per_asset_year}
                  r={5}
                  fill={STATUS.good}
                  stroke={INK.surface}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </ChartFrame>
  );
}
