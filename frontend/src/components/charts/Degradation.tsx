import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_PROPS, GRID_PROPS, INK, SERIES, STATUS } from "../../lib/viz";
import { ChartFrame, LegendKey, TooltipCard } from "./ChartFrame";

export type DegradationUnit = {
  unit_id: number;
  points: { cycle: number; true_rul: number; pred_rul: number }[];
};

/**
 * One engine's life: what the model thought was left against what actually was.
 *
 * Two series, so both are in the legend and both are direct-labelled at the
 * right-hand end. The alert line is an annotation in status colour, carrying a
 * text label rather than relying on the colour to mean "act here".
 */
export default function Degradation({
  unit,
  leadTime,
  modelName,
}: {
  unit: DegradationUnit;
  leadTime: number;
  modelName: string | null;
}) {
  const points = unit.points;
  const crossing = points.find((p) => p.pred_rul <= leadTime);
  const trueCrossing = points.find((p) => p.true_rul <= leadTime);
  const last = points[points.length - 1];

  return (
    <ChartFrame
      title={`Engine ${unit.unit_id} — predicted against true remaining life`}
      subtitle={`${modelName ?? "Champion model"} over ${points.length} recorded cycles. Both lines are capped at 125 cycles, the standard piecewise-linear C-MAPSS label.`}
      legend={
        <>
          <LegendKey colour={SERIES[0]} label="predicted RUL" />
          <LegendKey colour={SERIES[2]} label="true RUL" />
          <LegendKey colour={STATUS.warning} label={`alert at RUL ${leadTime}`} dashed />
        </>
      }
      footnote={
        crossing ? (
          <>
            The model first crosses the alert line at cycle <strong>{crossing.cycle}</strong>
            {trueCrossing && (
              <>
                {" "}
                against a true crossing at cycle <strong>{trueCrossing.cycle}</strong> — a{" "}
                {crossing.cycle - trueCrossing.cycle > 0 ? "late" : "early"} call by{" "}
                <strong>{Math.abs(crossing.cycle - trueCrossing.cycle)}</strong> cycles
              </>
            )}
            . Early is survivable; late is an unplanned failure.
          </>
        ) : (
          <>This engine never crossed the alert line within its recorded cycles.</>
        )
      }
      table={
        <table className="w-full text-[11px]">
          <thead className="text-muted sticky top-0 bg-[#111820]">
            <tr className="text-left">
              <th className="py-1 pr-4 font-medium">Cycle</th>
              <th className="py-1 pr-4 text-right font-medium">True RUL</th>
              <th className="py-1 text-right font-medium">Predicted RUL</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.cycle}>
                <td className="py-0.5 pr-4 tabular-nums">{p.cycle}</td>
                <td className="py-0.5 pr-4 text-right tabular-nums">{p.true_rul.toFixed(0)}</td>
                <td className="py-0.5 text-right tabular-nums">{p.pred_rul.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 70, bottom: 20, left: 8 }}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis
            dataKey="cycle"
            type="number"
            domain={["dataMin", "dataMax"]}
            label={{
              value: "Cycle",
              position: "insideBottom",
              offset: -12,
              fill: INK.muted,
              fontSize: 11,
            }}
            {...AXIS_PROPS}
          />
          <YAxis width={44} {...AXIS_PROPS} />
          <Tooltip
            cursor={{ stroke: INK.muted, strokeWidth: 1, strokeDasharray: "3 3" }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as (typeof points)[number];
              return (
                <TooltipCard
                  heading={`Cycle ${label}`}
                  rows={[
                    ["True RUL", p.true_rul.toFixed(0)],
                    ["Predicted RUL", p.pred_rul.toFixed(1)],
                    ["Error", (p.pred_rul - p.true_rul).toFixed(1)],
                  ]}
                />
              );
            }}
          />
          <ReferenceLine
            y={leadTime}
            stroke={STATUS.warning}
            strokeDasharray="4 3"
            strokeWidth={1.5}
          />
          {crossing && (
            <ReferenceLine x={crossing.cycle} stroke={STATUS.warning} strokeDasharray="2 4" strokeWidth={1} />
          )}
          <Line
            type="monotone"
            dataKey="true_rul"
            stroke={SERIES[2]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            label={({ index, x, y }: { index?: number; x?: number; y?: number }) =>
              index === points.length - 1 && x !== undefined && y !== undefined ? (
                // Both series end near zero, so the labels are split vertically
                // rather than stacked on the same pixel.
                <text x={x + 6} y={y - 8} fill={SERIES[2]} fontSize={10} dominantBaseline="middle">
                  true
                </text>
              ) : (
                <g />
              )
            }
          />
          <Line
            type="monotone"
            dataKey="pred_rul"
            stroke={SERIES[0]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            activeDot={{ r: 4, fill: SERIES[0], stroke: INK.surface, strokeWidth: 2 }}
            label={({ index, x, y }: { index?: number; x?: number; y?: number }) =>
              index === points.length - 1 && x !== undefined && y !== undefined ? (
                <text x={x + 6} y={y + 8} fill={SERIES[0]} fontSize={10} dominantBaseline="middle">
                  predicted
                </text>
              ) : (
                <g />
              )
            }
          />
          {last && <ReferenceLine x={last.cycle} stroke="transparent" />}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
