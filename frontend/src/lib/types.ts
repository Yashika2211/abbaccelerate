/** Shapes returned by the Kairos API. Kept in one place so pages cannot drift. */

export type Health = {
  status: "ok" | "degraded";
  version: string;
  env: string;
  uptime_s: number;
  components: { postgres: string; mlflow: string; llm: string; checkpointer: string };
};

export type BuiltinDataset = { key: string; name: string; description: string };

export type LeakageWarning = {
  column: string;
  severity: "certain" | "suspected";
  evidence: string;
  score: number;
};

export type ColumnProfile = {
  name: string;
  dtype: string;
  missing_fraction: number;
  n_unique: number;
  is_constant: boolean;
  is_identifier: boolean;
  is_numeric: boolean;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
};

export type DatasetProfile = {
  name: string;
  n_rows: number;
  n_columns: number;
  duplicate_rows: number;
  columns: ColumnProfile[];
  constant_columns: string[];
  identifier_columns: string[];
  candidate_group_columns: string[];
  candidate_time_columns: string[];
  proposed_task_type: string | null;
  proposed_target: string | null;
  class_balance: Record<string, number> | null;
  leakage_warnings: LeakageWarning[];
  documented_leakage_undetected: string[];
  most_informative_features: { column: string; mutual_information: number }[];
  recommended_drops: string[];
  findings: string[];
};

export type DatasetDetail = {
  dataset_id: string;
  name: string;
  citation: string;
  source: string;
  profile: DatasetProfile | null;
};

export type CostConfig = {
  c_unplanned_repair: number;
  c_planned_repair: number;
  c_inspection: number;
  downtime_rate_per_hour: number;
  unplanned_downtime_hours: number;
  planned_downtime_hours: number;
  c_secondary_damage: number;
  part_lead_time_cycles: number;
  value_per_remaining_cycle: number;
  technician_capacity_per_day: number;
  currency: string;
  cycles_per_year: number;
  observations_per_asset_year: number;
};

export type TimelineEvent = {
  sequence: number;
  node: string;
  status: "started" | "done" | "failed" | "skipped";
  summary: string;
  reasoning: string | null;
  duration_ms: number | null;
  created_at: string;
};

export type RunDetail = {
  run_id: string;
  status: string;
  task_type: string | null;
  awaiting_approval: boolean;
  cost_config: CostConfig | null;
  plan: Record<string, unknown> | null;
  findings: string[];
  impact: Impact | null;
  error: string | null;
  llm_used: boolean;
  n_trials: number;
};

export type LeaderboardRow = {
  model_id: string;
  model_name: string;
  operating_point: number;
  operating_point_kind: "threshold" | "lead_time";
  total_cost: number;
  cost_per_asset_year: number;
  cost_regret: number;
  metrics: Record<string, number>;
  degenerate: boolean;
  degenerate_reason: string | null;
};

export type Baseline = {
  key: string;
  name: string;
  description: string;
  total_cost: number;
  cost_per_asset_year: number;
};

export type Leaderboard = {
  run_id: string;
  rows: LeaderboardRow[];
  currency: string;
  elapsed_ms: number;
  empty_reason?: string;
  baselines: {
    baselines: Baseline[];
    best: string;
    kairos_wins: boolean;
    savings_per_asset_year: number;
    verdict: string;
    currency: string;
  } | null;
};

export type Impact = {
  champion: string;
  operating_point: number;
  operating_point_kind: string;
  cost_per_asset_year: number;
  currency: string;
  n_models_compared: number;
  llm_used: boolean;
  verdict?: string;
  savings_per_asset_year?: number;
  kairos_wins?: boolean;
  annualisation_note?: string;
  findings: string[];
  empty_reason?: string;
};

export type AgentEvent = {
  type: string;
  run_id?: string;
  node?: string;
  status?: string;
  summary?: string;
  reasoning?: string;
  duration_ms?: number;
  plan?: Record<string, unknown>;
  findings?: string[];
  model_name?: string;
  metrics?: Record<string, number>;
  ok?: boolean;
  fit_seconds?: number;
  message?: string;
  ts?: number;
};
