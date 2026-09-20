import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { num, pct } from "../../lib/format";
import { AXIS_PROPS, GRID_PROPS, INK, SERIES } from "../../lib/viz";
import { ChartFrame, LegendKey, TooltipCard } from "./ChartFrame";

export type ReliabilityBin = {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  observed_frequency: number;
  count: number;
};

export type CalibrationModel = {
  model_id: string;
  model_name: string;
  is_champion: boolean;
  method: string;
  brier_before: number;
  brier_after: number;
  ece_before: number;
  ece_after: number;
  bins: ReliabilityBin[];
  n_calibration_rows: number;
  n_calibration_positives: number;
};

/**
 * Reliability diagram: predicted probability against observed frequency.
 *
 * This is the answer to "why should I trust 0.71?". Without it, a cost-optimal
 * threshold is arithmetic on numbers that may not mean what they claim.
 *
 * One series plus a reference diagonal. The diagonal is an annotation, not data,
 * so it is drawn in ink rather than a series hue.
 */
export default function Reliability({ model }: { model: CalibrationModel }) {
  // A bin holding a handful of rows can sit at 0% or 100% purely by chance. Bubble
  // area already encodes count, but the sparse bins are also de-emphasised so they
  // cannot be misread as a calibration failure.
  const SPARSE = 20;
  const points = model.bins.map((b) => ({
    x: b.mean_predicted,
    y: b.observed_frequency,
    count: b.count,
    sparse: b.count < SPARSE,
    range: `${b.bin_lower.toFixed(1)}–${b.bin_upper.toFixed(1)}`,
  }));
  const sparseCount = points.filter((p) => p.sparse).length;
  const improved = model.ece_after <= model.ece_before;

  return (
    <ChartFrame
      title={`Reliability — ${model.model_name}`}
      subtitle={`Calibrated with ${model.method === "sigmoid" ? "Platt scaling" : model.method} on ${model.n_calibration_rows.toLocaleString()} validation rows carrying ${model.n_calibration_positives} positives. A point on the diagonal means the model's stated confidence matches how often it is actually right.`}
      legend={
        <>
          <LegendKey colour={SERIES[0]} label="observed frequency" />
          <LegendKey colour={INK.deemphasis} label={`fewer than ${SPARSE} rows`} />
          <LegendKey colour={INK.muted} label="perfect calibration" dashed />
        </>
      }
      footnote={
        <>
          Expected calibration error {num(model.ece_before, 4)} → <strong>{num(model.ece_after, 4)}</strong>{" "}
          after calibration{improved ? "" : " (calibration did not improve it here, and that is reported rather than hidden)"}.
          Brier {num(model.brier_before, 4)} → <strong>{num(model.brier_after, 4)}</strong>. An ECE of
          0.02 means that when this model says 71%, it is right about 69–73% of the time.
          {sparseCount > 0 && (
            <>
              {" "}
              {sparseCount} of {points.length} bins hold fewer than {SPARSE} rows and are greyed:
              at a {model.n_calibration_positives}-positive base rate they swing to 0% or 100% on
              one row, so read them as noise rather than miscalibration.
            </>
          )}
        </>
      }
      table={
        <table className="w-full text-[11px]">
          <thead className="text-muted sticky top-0 bg-[#111820]">
            <tr className="text-left">
              <th className="py-1 pr-4 font-medium">Bin</th>
              <th className="py-1 pr-4 text-right font-medium">Predicted</th>
              <th className="py-1 pr-4 text-right font-medium">Observed</th>
              <th className="py-1 text-right font-medium">Rows</th>
            </tr>
          </thead>
          <tbody>
            {model.bins.map((b) => (
              <tr key={b.bin_lower} className={b.count < SPARSE ? "text-muted" : ""}>
                <td className="py-0.5 pr-4 tabular-nums">
                  {b.bin_lower.toFixed(1)}–{b.bin_upper.toFixed(1)}
                </td>
                <td className="py-0.5 pr-4 text-right tabular-nums">{pct(b.mean_predicted, 1)}</td>
                <td className="py-0.5 pr-4 text-right tabular-nums">
                  {pct(b.observed_frequency, 1)}
                </td>
                <td className="py-0.5 text-right tabular-nums">{b.count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
          <CartesianGrid {...GRID_PROPS} vertical />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            tickFormatter={(v: number) => pct(v, 0)}
            label={{
              value: "Predicted probability",
              position: "insideBottom",
              offset: -12,
              fill: INK.muted,
              fontSize: 11,
            }}
            {...AXIS_PROPS}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, 1]}
            tickFormatter={(v: number) => pct(v, 0)}
            width={48}
            {...AXIS_PROPS}
          />
          {/* Bubble size carries how many rows a bin holds — a bin of 12 rows
              should not read as confidently as a bin of 4,000. */}
          <ZAxis type="number" dataKey="count" range={[40, 320]} />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke={INK.muted}
            strokeDasharray="3 4"
            strokeWidth={1}
            ifOverflow="extendDomain"
          />
          <Tooltip
            cursor={{ stroke: INK.muted, strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as (typeof points)[number];
              return (
                <TooltipCard
                  heading={`Bin ${p.range}`}
                  rows={[
                    ["Model says", pct(p.x, 1)],
                    ["Actually happens", pct(p.y, 1)],
                    ["Rows in bin", p.count.toLocaleString()],
                  ]}
                />
              );
            }}
          />
          <Scatter data={points} isAnimationActive={false}>
            {points.map((p) => (
              <Cell
                key={`${p.x}-${p.count}`}
                fill={p.sparse ? INK.deemphasis : SERIES[0]}
                stroke={INK.surface}
                strokeWidth={2}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
