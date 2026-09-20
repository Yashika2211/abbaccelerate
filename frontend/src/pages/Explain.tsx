import { useEffect, useState } from "react";
import { charts } from "../lib/api";
import { useSession } from "../lib/session";
import { Card, EmptyState } from "../components/ui";
import CostCurve, { type CostCurveData } from "../components/charts/CostCurve";
import Degradation, { type DegradationUnit } from "../components/charts/Degradation";
import Reliability, { type CalibrationModel } from "../components/charts/Reliability";
import ShapBars, { type ShapFeature } from "../components/charts/ShapBars";

type Explanations = { model_name: string | null; global: ShapFeature[]; n_rows_explained?: number; empty_reason?: string };
type Calibration = { models: CalibrationModel[]; empty_reason?: string };
type Degradations = { units: DegradationUnit[]; lead_time: number; model_name: string | null; empty_reason?: string };

/** Anything absent renders as the reason it is absent — never as a blank chart. */
function Unavailable({ title, reason }: { title: string; reason?: string }) {
  return (
    <Card title={title}>
      <EmptyState title="Not available for this run" detail={reason} />
    </Card>
  );
}

export default function Explain() {
  const [session] = useSession();
  const [curve, setCurve] = useState<CostCurveData | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [shap, setShap] = useState<Explanations | null>(null);
  const [degradation, setDegradation] = useState<Degradations | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!session.runId) return;
    setLoading(true);
    const id = session.runId;
    Promise.allSettled([
      charts.costCurve(id),
      charts.calibration(id),
      charts.explain(id),
      charts.degradation(id),
    ])
      .then(([c, cal, s, d]) => {
        if (c.status === "fulfilled") setCurve(c.value as unknown as CostCurveData);
        if (cal.status === "fulfilled") setCalibration(cal.value as unknown as Calibration);
        if (s.status === "fulfilled") setShap(s.value as unknown as Explanations);
        if (d.status === "fulfilled") setDegradation(d.value as unknown as Degradations);
      })
      .finally(() => setLoading(false));
  }, [session.runId]);

  if (!session.runId) {
    return (
      <EmptyState
        title="No run selected"
        detail="Start a run to see how its decision was reached."
      />
    );
  }
  if (loading) return <p className="text-muted text-xs">loading charts…</p>;

  const champion = calibration?.models?.find((m) => m.is_champion) ?? calibration?.models?.[0];

  return (
    <div className="space-y-5">
      {curve && curve.points?.length ? (
        <Card>
          <CostCurve data={curve} />
          {curve.degenerate && curve.degenerate_reason && (
            <p className="border-alarm/30 bg-alarm/10 text-alarm mt-3 rounded border px-3 py-2 text-xs">
              {curve.degenerate_reason}
            </p>
          )}
          {curve.calibration_warning && (
            <p className="mt-3 rounded border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-300">
              {curve.calibration_warning}
            </p>
          )}
        </Card>
      ) : (
        <Unavailable title="Cost curve" reason={curve?.empty_reason} />
      )}

      {shap?.global?.length ? (
        <Card>
          <ShapBars
            features={shap.global}
            modelName={shap.model_name}
            nRows={shap.n_rows_explained}
          />
        </Card>
      ) : (
        <Unavailable title="Feature attribution" reason={shap?.empty_reason} />
      )}

      {champion ? (
        <Card>
          <Reliability model={champion} />
        </Card>
      ) : (
        <Unavailable title="Calibration" reason={calibration?.empty_reason} />
      )}

      {degradation?.units?.length ? (
        <div className="grid gap-5 lg:grid-cols-2">
          {degradation.units.slice(0, 4).map((unit) => (
            <Card key={unit.unit_id}>
              <Degradation
                unit={unit}
                leadTime={degradation.lead_time}
                modelName={degradation.model_name}
              />
            </Card>
          ))}
        </div>
      ) : (
        <Unavailable title="Degradation trajectories" reason={degradation?.empty_reason} />
      )}
    </div>
  );
}
