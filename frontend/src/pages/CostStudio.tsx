import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { money, moneyExact, pct } from "../lib/format";
import { useSession } from "../lib/session";
import type { CostConfig } from "../lib/types";
import { Button, Card, EmptyState, ErrorNote, Stat } from "../components/ui";

/** Matches backend presets; a plant manager edits these, they are not claims. */
const TURBOFAN: CostConfig = {
  c_unplanned_repair: 800_000,
  c_planned_repair: 120_000,
  c_inspection: 15_000,
  downtime_rate_per_hour: 45_000,
  unplanned_downtime_hours: 12,
  planned_downtime_hours: 3,
  c_secondary_damage: 0,
  part_lead_time_cycles: 5,
  value_per_remaining_cycle: 2_000,
  technician_capacity_per_day: 3,
  currency: "INR",
  cycles_per_year: 500,
  observations_per_asset_year: 10_000,
};

const MILLING: CostConfig = {
  ...TURBOFAN,
  c_unplanned_repair: 25_000,
  c_planned_repair: 6_000,
  c_inspection: 1_200,
  downtime_rate_per_hour: 9_000,
  unplanned_downtime_hours: 3,
  planned_downtime_hours: 0.5,
  part_lead_time_cycles: 0,
  value_per_remaining_cycle: 0,
  technician_capacity_per_day: 6,
};

type FieldSpec = {
  key: keyof CostConfig;
  label: string;
  hint: string;
  step?: number;
  money?: boolean;
};

const FIELDS: { group: string; fields: FieldSpec[] }[] = [
  {
    group: "Cost of failing",
    fields: [
      {
        key: "c_unplanned_repair",
        label: "Unplanned repair",
        hint: "Emergency parts and labour when it breaks without warning.",
        money: true,
      },
      {
        key: "unplanned_downtime_hours",
        label: "Unplanned downtime (h)",
        hint: "Typically 3–10× a planned stop.",
      },
      {
        key: "c_secondary_damage",
        label: "Secondary damage",
        hint: "Collateral from a catastrophic failure.",
        money: true,
      },
    ],
  },
  {
    group: "Cost of acting",
    fields: [
      {
        key: "c_planned_repair",
        label: "Planned repair",
        hint: "Scheduled parts and labour in a normal maintenance slot.",
        money: true,
      },
      { key: "planned_downtime_hours", label: "Planned downtime (h)", hint: "", step: 0.5 },
      {
        key: "c_inspection",
        label: "Inspection call-out",
        hint: "Technician dispatched and nothing found. This is the price of a false alarm.",
        money: true,
      },
    ],
  },
  {
    group: "Production and logistics",
    fields: [
      {
        key: "downtime_rate_per_hour",
        label: "Downtime rate / hour",
        hint: "Lost production while the asset is stopped.",
        money: true,
      },
      {
        key: "part_lead_time_cycles",
        label: "Spare-part lead time (cycles)",
        hint: "An alert is worthless if the part cannot arrive before failure.",
      },
      {
        key: "value_per_remaining_cycle",
        label: "Value per remaining cycle",
        hint: "What you throw away by replacing a component early.",
        money: true,
      },
      {
        key: "technician_capacity_per_day",
        label: "Technicians / day",
        hint: "How many work orders can actually be serviced.",
      },
    ],
  },
  {
    group: "Annualisation",
    fields: [
      {
        key: "cycles_per_year",
        label: "Cycles / asset / year",
        hint: "RUL work. Neither dataset carries a calendar, so this is stated, not assumed silently.",
      },
      {
        key: "observations_per_asset_year",
        label: "Observations / asset-year",
        hint: "Classification work. 10,000 rows ≈ one machine-year.",
      },
    ],
  },
];

export default function CostStudio() {
  const [session, update] = useSession();
  const [cfg, setCfg] = useState<CostConfig>(session.costConfig ?? TURBOFAN);
  const [budget, setBudget] = useState(180);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!session.datasetId) return;
    api
      .datasetProfile(session.datasetId)
      .then((d) => {
        if (session.costConfig) return;
        const isClassification = d.profile?.proposed_task_type === "binary_classification";
        setCfg(isClassification ? MILLING : TURBOFAN);
      })
      .catch(() => {});
  }, [session.datasetId, session.costConfig]);

  /** The three numbers that explain any cost config at a glance. */
  const derived = useMemo(() => {
    const miss =
      cfg.c_unplanned_repair +
      cfg.unplanned_downtime_hours * cfg.downtime_rate_per_hour +
      cfg.c_secondary_damage;
    const act = cfg.c_planned_repair + cfg.planned_downtime_hours * cfg.downtime_rate_per_hour;
    const falseAlarm = cfg.c_inspection;
    const denominator = falseAlarm + miss - act;
    return {
      miss,
      act,
      falseAlarm,
      ratio: falseAlarm > 0 ? miss / falseAlarm : Infinity,
      bayes: denominator > 0 ? Math.min(Math.max(falseAlarm / denominator, 0), 1) : 0,
    };
  }, [cfg]);

  function set(key: keyof CostConfig, value: string) {
    const parsed = Number(value);
    if (Number.isNaN(parsed) || parsed < 0) return;
    setCfg((prev) => ({ ...prev, [key]: parsed }));
  }

  async function startRun() {
    if (!session.datasetId) return;
    setStarting(true);
    setError(null);
    try {
      update({ costConfig: cfg });
      const created = await api.createRun(session.datasetId, cfg, budget);
      update({ runId: created.run_id });
      navigate("/run");
    } catch (e) {
      setError(String(e));
      setStarting(false);
    }
  }

  if (!session.datasetId) {
    return (
      <EmptyState
        title="Pick a dataset first"
        detail="Plant economics are meaningful per asset, so Kairos suggests a starting profile once it knows what you are maintaining."
      />
    );
  }

  return (
    <div className="space-y-5">
      <Card
        title="Cost Studio"
        subtitle="Your economics, not ours. Everything downstream is recomputed from these numbers."
        right={
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => setCfg(TURBOFAN)}>
              Turbofan
            </Button>
            <Button variant="ghost" onClick={() => setCfg(MILLING)}>
              Milling
            </Button>
          </div>
        }
      >
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Cost of a miss" value={money(derived.miss)} tone="bad" />
          <Stat label="Cost of acting" value={money(derived.act)} />
          <Stat label="Cost of a false alarm" value={money(derived.falseAlarm)} />
          <Stat
            label="Break-even probability"
            value={Number.isFinite(derived.ratio) ? pct(derived.bayes, 2) : "0%"}
            hint={
              Number.isFinite(derived.ratio)
                ? `${Math.round(derived.ratio)}:1 consequence ratio`
                : "inspection is free"
            }
          />
        </dl>
        <p className="text-muted mt-3 text-[11px]">
          Alert on anything more likely than the break-even probability. That figure is the
          closed-form optimum for these costs; Kairos also sweeps the threshold empirically and
          shows you both, so you can see whether the model's probabilities agree with the theory.
        </p>
      </Card>

      {FIELDS.map((section) => (
        <Card key={section.group} title={section.group}>
          <div className="grid gap-4 sm:grid-cols-3">
            {section.fields.map((f) => (
              <label key={String(f.key)} className="block">
                <span className="text-muted text-[11px] tracking-wide uppercase">{f.label}</span>
                <input
                  type="number"
                  min={0}
                  step={f.step ?? 1}
                  value={Number(cfg[f.key])}
                  onChange={(e) => set(f.key, e.target.value)}
                  className="border-edge focus:border-accent mt-1 w-full rounded border bg-black/30 px-2 py-1.5 text-sm tabular-nums outline-none"
                />
                {f.money && (
                  <span className="text-muted mt-0.5 block text-[11px]">
                    {moneyExact(Number(cfg[f.key]))}
                  </span>
                )}
                {f.hint && <span className="text-muted mt-0.5 block text-[11px]">{f.hint}</span>}
              </label>
            ))}
          </div>
        </Card>
      ))}

      <Card title="Run">
        <div className="flex flex-wrap items-end gap-4">
          <label className="block">
            <span className="text-muted text-[11px] tracking-wide uppercase">
              Training budget (s)
            </span>
            <input
              type="number"
              min={30}
              max={1800}
              step={30}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="border-edge focus:border-accent mt-1 w-32 rounded border bg-black/30 px-2 py-1.5 text-sm tabular-nums outline-none"
            />
          </label>
          <Button onClick={startRun} disabled={starting}>
            {starting ? "starting…" : "Start run →"}
          </Button>
          <p className="text-muted text-[11px]">
            The agent profiles, plans, and then stops for your approval before it trains anything.
          </p>
        </div>
        <ErrorNote error={error} />
      </Card>
    </div>
  );
}
