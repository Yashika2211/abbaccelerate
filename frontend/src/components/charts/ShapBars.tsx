import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { num } from "../../lib/format";
import { INK, sequentialStep } from "../../lib/viz";
import { ChartFrame, TooltipCard } from "./ChartFrame";

export type ShapFeature = { feature: string; mean_abs_shap: number };

/**
 * Global feature importance. Magnitude low→high, so one hue light→dark rather
 * than a categorical palette — these bars are one quantity, not eight identities.
 */
export default function ShapBars({
  features,
  modelName,
  nRows,
  limit = 12,
}: {
  features: ShapFeature[];
  modelName: string | null;
  nRows?: number;
  limit?: number;
}) {
  const data = features.slice(0, limit);

  return (
    <ChartFrame
      title="What drives the prediction"
      subtitle={`Mean absolute SHAP value per feature${modelName ? ` for ${modelName}` : ""}${nRows ? `, over ${nRows.toLocaleString()} held-out rows` : ""}. Computed with TreeExplainer — Kairos shows real attributions or none at all.`}
      footnote="Magnitude only: this ranks how much a feature moves the prediction, not which direction. Per-asset direction appears on each work order."
      table={
        <table className="w-full text-[11px]">
          <thead className="text-muted sticky top-0 bg-[#111820]">
            <tr className="text-left">
              <th className="py-1 pr-4 font-medium">Feature</th>
              <th className="py-1 text-right font-medium">Mean |SHAP|</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f) => (
              <tr key={f.feature}>
                <td className="py-0.5 pr-4 font-mono">{f.feature}</td>
                <td className="py-0.5 text-right tabular-nums">{num(f.mean_abs_shap, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
      height={Math.max(220, data.length * 26 + 40)}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
          barCategoryGap={4}
        >
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="feature"
            width={190}
            tick={{ fill: INK.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const f = payload[0].payload as ShapFeature;
              return (
                <TooltipCard
                  heading={f.feature}
                  rows={[["Mean |SHAP|", num(f.mean_abs_shap, 4)]]}
                />
              );
            }}
          />
          <Bar
            dataKey="mean_abs_shap"
            radius={[0, 4, 4, 0]}
            isAnimationActive={false}
            label={{
              position: "right",
              fill: INK.muted,
              fontSize: 10,
              formatter: (v: number) => num(v, 3),
            }}
          >
            {data.map((f, i) => (
              <Cell key={f.feature} fill={sequentialStep(i, data.length)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
