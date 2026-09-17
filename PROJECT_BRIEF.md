# KAIROS — Master Build Prompt

> Kept in the repo so it can be re-read after a context reset.

---

## 0. Role

Lead engineer building a hackathon project end-to-end in **24–36 hours**. Work in **phases with hard time boxes** (Section 10). At the end of each phase: stop, run the thing, report what works and what doesn't, wait for go-ahead.

Optimize for **a working demo that survives a judge's questions**, not for feature count. A broken feature is worth less than zero.

---

## 1. What we are building

**Kairos** — *the right moment to act.*

An agentic AutoML + MLOps platform for industrial predictive maintenance. An engineer uploads sensor data; an AI agent profiles it, picks the task, engineers features, trains and compares models, explains predictions, and deploys an inference endpoint.

That much is the theme. **Every other team will build that.** Here is why ours wins:

> **Every other predictive-maintenance tool stops at a probability. Kairos stops at a decision, in rupees.**

A model that says `P(failure) = 0.71` is useless to a plant manager. Kairos converts every prediction into **an action, a date, and a cost**, using the plant's own economics. This changes what the platform optimizes, what the leaderboard ranks by, and what the dashboard shows.

The headline demo moment:

> "XGBoost has 0.02 higher AUC than LightGBM. It also costs ₹4.2 lakh more per year, because on this asset a missed failure costs 40× a false alarm and XGBoost trades recall for precision. Kairos picks LightGBM. Here is the math."

---

## 2. The non-negotiable differentiator: the Decision Layer

Build first. Protect at all costs. If we run out of time we ship an ugly UI with a perfect Decision Layer, never the reverse.

### 2.1 Cost-optimal thresholding
Never 0.5. Never max-F1. Sweep the threshold across `[0, 1]` on the **validation** set, compute expected cost at each point, pick the argmin, then report performance on the **held-out test** set at that frozen threshold. Produce the cost-vs-threshold curve with the optimum marked.

### 2.2 Calibration
Cost-optimal decisions are only valid if probabilities mean what they say. Calibrate every classifier (isotonic on validation; Platt for small data) and render a **reliability diagram** alongside the cost curve.

### 2.3 Cost-ranked leaderboard
Primary sort key is **expected annual cost**, not AUC/RMSE. AUC, F1, RMSE are secondary columns. Include a **cost regret** column: `model_cost − best_model_cost`. Highlight explicitly whenever the best-AUC model is not the best-cost model.

### 2.4 Counterfactual what-if simulator
Change the economics (cost of unplanned failure, planned repair, inspection call-out, downtime rate, spare-part lead time, technician capacity) and the **entire decision surface recomputes live** — new optimal threshold, new work order queue, new leaderboard ordering, new savings. No retraining; pure post-hoc simulation over cached predictions, **under 300 ms**.

### 2.5 Honest baselines
Every savings claim is stated **relative to named baselines**, computed on the same test set:

1. **Run-to-failure** — never intervene.
2. **Fixed-interval preventive** — intervene every *k* cycles, with *k* itself optimized. Do not straw-man it.
3. **Model at threshold 0.5** — the naive ML approach.
4. **Kairos** — model at cost-optimal threshold.

Display all four as bars. If Kairos loses to optimized fixed-interval, **show it**.

---

## 3. Cost model — exact math

`backend/kairos/decision/cost.py`, **pure functions, no ML dependencies**, unit-tested before any model is trained.

### 3.1 Config
```python
@dataclass(frozen=True)
class CostConfig:
    currency: str = "INR"
    c_unplanned_repair: float          # emergency parts + labour
    c_planned_repair: float            # scheduled parts + labour
    c_inspection: float                # technician dispatched, nothing found
    downtime_rate_per_hour: float      # lost production
    unplanned_downtime_hours: float    # typically 3-10x planned
    planned_downtime_hours: float
    c_secondary_damage: float = 0.0    # collateral from catastrophic failure
    part_lead_time_cycles: int = 0     # alert is useless if it lands too late
    value_per_remaining_cycle: float = 0.0  # cost of scrapping good life
    technician_capacity_per_day: int = 999  # scheduling constraint
```

### 3.2 Classification cost matrix (per asset-event)
```
TN -> 0
FP -> c_inspection
TP -> c_planned_repair + planned_downtime_hours * downtime_rate_per_hour
FN -> c_unplanned_repair + unplanned_downtime_hours * downtime_rate_per_hour + c_secondary_damage
```
`expected_cost(y_true, y_pred, cfg) = Σ cost_matrix[y_true_i][y_pred_i]`

Report total cost **and cost per asset per year**, using the dataset's event rate to annualize. State the annualization assumption in the UI, in words.

### 3.3 RUL (regression) policy cost
Decision variable is **lead time `L`**: intervene at the first cycle `t` where predicted RUL ≤ `L`.

For each unit with true failure cycle `T`:
```
t_alert = min{ t : pred_RUL(t) <= L }, or ∞ if never
t_action = t_alert + part_lead_time_cycles

if t_action < T:      # caught it in time
    cost = c_planned_repair
         + planned_downtime_hours * downtime_rate_per_hour
         + (T - t_action) * value_per_remaining_cycle     # life thrown away
else:                 # too late, or never alerted
    cost = c_unplanned_repair
         + unplanned_downtime_hours * downtime_rate_per_hour
         + c_secondary_damage
```
Sweep `L ∈ [0, 60]`, plot total fleet cost vs `L`, mark the optimum. The curve is **U-shaped** — hero chart.

### 3.4 Risk-aware RUL via quantile regression
LightGBM `objective="quantile"` at `alpha ∈ {0.1, 0.5, 0.9}`. Decide on the **pessimistic (0.1) quantile**; expose a risk-quantile slider. Show the prediction interval as a band on the degradation chart. Let the cost engine pick the quantile that minimizes cost — a second, independent decision variable.

### 3.5 Scheduling
Given the alert set and `technician_capacity_per_day`, produce a prioritized work order queue: sort by `expected_cost_avoided / days_until_deadline`, greedily fill each day's capacity, flag assets that **cannot** be serviced in time. Greedy, not an LP solver.

---

## 4. Tech stack (pinned)

| Layer | Choice |
|---|---|
| API | FastAPI + Pydantic v2 + Uvicorn |
| Agent | LangGraph (StateGraph, Postgres checkpointer) |
| LLM | Claude via `anthropic` SDK; model + key from env, never hardcoded |
| ML | scikit-learn, LightGBM, XGBoost, pandas, numpy |
| Explainability | SHAP (TreeExplainer only) |
| Tracking | MLflow (server in Docker, Postgres backend, local artifact store) |
| DB | PostgreSQL 16 + SQLModel |
| Frontend | Vite + React + TypeScript + Tailwind + shadcn/ui + Recharts |
| Realtime | Server-Sent Events (**not** WebSockets) |
| Orchestration | Docker Compose (**no** Kubernetes, **no** Celery, **no** Redis) |
| Tests | pytest |

Background jobs run in FastAPI `BackgroundTasks` with progress pushed over SSE.

---

## 5. Repo structure

```
kairos/
  docker-compose.yml
  Makefile                      # make up / make seed / make demo / make test
  PROJECT_BRIEF.md
  README.md
  backend/
    kairos/
      main.py
      api/                      # routers, one per resource
      agent/
        graph.py                # LangGraph StateGraph
        state.py                # typed agent state
        nodes/                  # one file per node
        prompts/                # LLM prompts as .md, never inline f-strings
      data/
        loaders.py              # C-MAPSS, AI4I, generic CSV
        profiler.py
        splits.py               # group-aware + temporal splitting
        features.py
      ml/
        registry.py             # candidate model zoo
        train.py
        evaluate.py
        calibrate.py
        explain.py              # SHAP
      decision/
        cost.py                 # <- Section 3, pure functions
        policy.py               # threshold / lead-time optimization
        simulate.py             # what-if
        schedule.py             # work order queue
        baselines.py
      serving/
        deploy.py               # promote to champion, build endpoint
      db/
        models.py               # SQLModel tables
      tests/
        test_cost.py            # <- write this FIRST
        test_splits.py          # leakage guards
        test_grounding.py       # LLM narrative grounding
  frontend/
    src/
      pages/                    # Upload, Cost, Run, Leaderboard, Explain, Decisions, Deploy, Impact
      components/charts/
      lib/api.ts
  data/
    raw/                        # gitignored
    samples/                    # small vendored slices, COMMITTED
  scripts/
    fetch_data.py
    seed_demo.py
```

---

## 6. Data contract — and the leakage traps

Two real, citable public datasets. `scripts/fetch_data.py` downloads them; a small slice is vendored into `data/samples/` and committed; a physics-based synthetic generator is the fallback. **The demo must never depend on the venue WiFi.**

### 6.1 NASA C-MAPSS (turbofan run-to-failure) → RUL regression
- Columns: `unit_id`, `cycle`, 3 operational settings, 21 sensors. **FD001** for the demo; FD002/003/004 loadable.
- **Trap 1 — split leakage.** Split by `unit_id` with `GroupShuffleSplit`. A random row split produces a fraudulent R² near 0.99. Test asserts zero `unit_id` overlap across splits.
- **Trap 2 — RUL labels.** Apply the **piecewise-linear cap at 125 cycles** (`RUL = min(T - t, 125)`), and say so in the UI.
- **Trap 3 — dead sensors.** In FD001 sensors 1, 5, 6, 10, 16, 18, 19 are constant or near-constant. The profiler detects and drops zero-variance columns — and **shows that it did**.
- **Features:** per-unit rolling mean/std/min/max over windows {5, 10, 20}, first-difference, cumulative cycle count, per-sensor linear-trend slope. All computed **grouped by `unit_id`**.
- Test-set RUL ground truth lives in `RUL_FD001.txt`; test trajectories are truncated before failure. Handle explicitly.

### 6.2 AI4I 2020 (UCI) → binary classification
- 10,000 rows, 3.39% positive rate.
- **Trap 4 — target leakage.** Drop `UDI`, `Product ID`, and **`TWF`, `HDF`, `PWF`, `OSF`, `RNF`** — these five are components of the `Machine failure` target. Build a leakage detector that flags high mutual information against the target and **surface the warning in the UI as an agent finding**.
- **Trap 5 — imbalance.** Never report accuracy. Use PR-AUC, recall at fixed precision, expected cost. Use `scale_pos_weight` / class weights, not SMOTE.
- **Engineered features:** `power = torque × rotational_speed × 2π/60`, `temp_delta = process_temp − air_temp`, `tool_wear × torque`.

### 6.3 Generic CSV upload
Profiled for: dtype inference, missingness, cardinality, zero-variance, duplicate rows, class balance, candidate time and group columns, leakage suspicion, drift-prone columns. The agent **proposes** task type and target and asks for confirmation.

---

## 7. The agent (LangGraph)

The agent must **branch on what it observes** and **re-plan on failure**.

### State
```python
class KairosState(TypedDict):
    run_id: str
    dataset_id: str
    profile: dict | None
    task_type: Literal["rul_regression","binary_classification"] | None
    plan: dict | None              # preprocessing + candidate models + budget
    trials: list[dict]             # every trained candidate
    best: dict | None
    cost_analysis: dict | None
    explanations: dict | None
    decisions: list[dict]
    findings: list[str]            # human-readable agent observations
    replan_count: int
    awaiting_approval: bool
    error: str | None
```

### Nodes
1. `profile_dataset` — statistical profiling (deterministic, no LLM).
2. `diagnose` — LLM reads the profile JSON and emits findings, task type, target, leakage warnings, preprocessing plan. **Structured output only**, validated by Pydantic; retry on validation failure.
3. `human_checkpoint` — LangGraph `interrupt`. Real HITL with a checkpointer, not a fake modal.
4. `engineer_features` — executes the approved plan.
5. `train_candidates` — time-budgeted loop over the model zoo with group-aware CV, every trial logged to MLflow.
6. `evaluate` — metrics + calibration.
7. `decide` — the Decision Layer: thresholds, cost curves, baselines, ranking.
8. `explain` — SHAP global + per-instance for the top-N riskiest assets.
9. `narrate` — LLM writes root-cause summaries and work orders **from a JSON payload of computed numbers**.
10. `report` — executive impact summary.

### Conditional edges
- After `evaluate`: if the best model fails the quality gate (PR-AUC < 0.6 classification, or RMSE worse than predict-the-mean for RUL) **and** `replan_count < 2` → route back to `diagnose` with the failure reason injected. Log the retry as a visible finding.
- After `decide`: if the cost-optimal threshold lands at an extreme (< 0.02 or > 0.98) → warn that the economics are degenerate and ML adds nothing here. **Honesty as a feature.**
- On any node exception → `recover` node: log, degrade gracefully, continue with a reduced plan.

Every node appends to an **audit trail** table (timestamp, node, inputs hash, decision, reasoning, duration) rendered as a live timeline. Stream every state transition over SSE.

---

## 8. API surface

```
POST   /api/datasets                  upload CSV / select builtin -> dataset_id
GET    /api/datasets/{id}/profile
POST   /api/runs                      {dataset_id, cost_config, time_budget_s} -> run_id
GET    /api/runs/{id}/stream          SSE: agent events, trials, progress
POST   /api/runs/{id}/approve         resume from human_checkpoint
GET    /api/runs/{id}/leaderboard     cost-ranked
GET    /api/runs/{id}/cost-curve
GET    /api/runs/{id}/baselines
POST   /api/runs/{id}/simulate        new CostConfig -> recomputed decisions (<300ms)
GET    /api/models/{id}/explain       global SHAP
POST   /api/models/{id}/explain/local {features} -> per-instance SHAP + narrative
POST   /api/models/{id}/promote       -> MLflow champion alias
POST   /api/predict/{model_alias}     live inference
GET    /api/runs/{id}/workorders      scheduled queue
GET    /api/runs/{id}/impact          executive summary
GET    /api/health
```

Every prediction response carries: `prediction`, `calibrated_confidence`, `interval` (RUL), `recommended_action`, `act_by_date`, `expected_cost_if_ignored`, `expected_cost_if_acted`, `top_3_drivers` (SHAP), `model_version`.

---

## 9. LLM grounding rules

The LLM writes prose. It does **not** produce numbers.

- Every narration prompt receives a JSON payload of already-computed values. The prompt says: use only the values provided; if a value is absent, say it is unavailable.
- `tests/test_grounding.py` asserts every feature name in a generated work order exists in the model's feature list, and every numeric token in the output appears in the input payload (allowing rounding). Fail on hallucinated features or invented figures.
- If the Anthropic API key is missing or the call fails, fall back to **deterministic template narration**.

---

## 10. Build order with time boxes

| Phase | Hours | Deliverable | Done when |
|---|---|---|---|
| 0 | 0–2 | Scaffold, docker-compose, Makefile, health check | `make up` works from a clean clone |
| 1 | 2–5 | **Cost engine + tests**, zero ML | `pytest tests/test_cost.py` green, ≥12 cases incl. degenerate economics |
| 2 | 5–8 | Data loaders, profiler, leakage-safe splits + split tests | Both datasets load; `test_splits.py` proves no unit overlap and no leaked AI4I columns |
| 3 | 8–12 | Training core: 4 candidates × 2 task types, group CV, MLflow, calibration | A run produces a cost-ranked leaderboard **from the CLI** |
| 4 | 12–15 | LangGraph agent + FastAPI + SSE, HITL interrupt | **FIRST END-TO-END DEMO — must exist by hour 15.** |
| 5 | 15–21 | Frontend: Upload → Cost Studio → Run timeline → Leaderboard | Clickable path through the whole flow |
| 6 | 21–25 | SHAP, reliability diagram, cost curve, quantile bands, degradation chart | Explain page renders real values |
| 7 | 25–28 | Decisions page: work order queue, scheduling, grounded narratives | Work orders cite real SHAP drivers |
| 8 | 28–31 | What-if simulator + baselines + Impact page | Sliders recompute in <300 ms; four baselines charted |
| 9 | 31–33 | One-click deploy: promote to champion, live `/predict`, curl + Dockerfile | A prediction served from the promoted model in the UI |
| 10 | 33–35 | `make seed`, README + architecture diagram, demo rehearsal | Full demo from cold start in under 4 minutes |
| 11 | 35–36 | **Freeze.** Backup screen capture. No new code. | Video exists |

**Cut list, in order:** multi-dataset FD002-4 → Dockerfile generation → drift monitoring → auth → dark mode → animations.
**Never cut:** the cost engine, the leaderboard's cost ranking, the cost curve chart, the baselines.

---

## 11. Guardrails

- **No fabricated numbers anywhere.** No placeholder charts, no `Math.random()`, no lorem ipsum. Every number traces to a computed value. If data is missing, render an empty state.
- **No mocked SHAP.**
- **No target leakage.** Write the tests.
- **No accuracy metric on imbalanced data.**
- **No unhandled exceptions in the demo path.**
- **No secrets in the repo.** `.env.example` only.
- **No Kubernetes, no microservices, no auth, no user accounts.**
- **Don't gold-plate the UI before hour 15.**
- **Commit at every phase boundary.**
- **If unsure about intent, ask.**
- **If something is broken, say it's broken.**

---

## 12. Demo script (3 minutes)

1. **(20s)** Upload C-MAPSS FD001. Profiler flags 7 dead sensors, proposes RUL regression.
2. **(20s)** Cost Studio: unplanned failure ₹8L, planned repair ₹1.2L, downtime ₹45k/hour, part lead time 5 cycles.
3. **(40s)** Run. Live agent timeline: plan → approve → features → 4 models → first attempt underperforms → **agent re-plans** → converges.
4. **(30s)** Leaderboard. XGBoost tops AUC. **LightGBM tops the ranking.** Cost regret column.
5. **(30s)** The U-curve. Optimal lead time is 23 cycles, not the 40 the manual says.
6. **(20s)** Decisions. Nine engines flagged, scheduled against 3 technicians/day. Two flagged **unserviceable in time**.
7. **(20s)** Drag the "cost of unplanned failure" slider. Everything recomputes live.
8. **(20s)** Promote to champion → live `/predict` call.
9. **(closing)** *"Every other tool here tells you a machine will fail. Kairos tells you when to send someone, and what it costs if you don't."*

---

## 13. Working agreement

- Read the brief back in five bullets; push back before writing code.
- Propose the Phase 0 file tree and wait.
- Prefer boring, working code over clever code.
- At the end of a phase: run it, paste the actual output, state what's working and what isn't, stop.
