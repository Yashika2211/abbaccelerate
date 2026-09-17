# Kairos — *the right moment to act*

Agentic AutoML + MLOps for industrial predictive maintenance.

> Every other predictive-maintenance tool stops at a probability.
> **Kairos stops at a decision, in rupees.**

A model that says `P(failure) = 0.71` is useless to a plant manager. Kairos
converts every prediction into **an action, a date, and a cost**, using the
plant's own economics — which changes what the platform optimizes, what the
leaderboard ranks by, and what the dashboard shows.

Full spec: [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

---

## Quickstart

```bash
cp .env.example .env     # optional: add ANTHROPIC_API_KEY
make up                  # postgres + mlflow + api + web
make health              # should print {"status":"ok",...}
```

| Service | URL |
|---|---|
| Web | http://localhost:5173 |
| API | http://localhost:8010/api/health |
| API docs | http://localhost:8010/docs |
| MLflow | http://localhost:5001 |
| Postgres | `localhost:5434` |

Host ports are overridable in `.env` (`KAIROS_API_PORT`, `KAIROS_WEB_PORT`,
`KAIROS_PG_PORT`, `KAIROS_MLFLOW_PORT`) for when a venue laptop already has
something on one of them.

`make down` stops everything, `make nuke` also drops the volumes.

---

## Standing assumptions

These are stated here because several headline numbers depend on them, and a
number whose denominator is invisible is not a number.

| Assumption | Value | Why |
|---|---|---|
| C-MAPSS annualization | 500 cycles / engine / year | Cycles are flights; the dataset carries no calendar. Editable in Cost Studio. |
| AI4I annualization | 10,000 rows ≈ 1 machine-year | The dataset carries no timespan. Editable in Cost Studio. |
| RUL label cap | 125 cycles, piecewise-linear | Standard for C-MAPSS; true RUL is only linear near end of life. |
| LLM | Optional | Without `ANTHROPIC_API_KEY`, diagnosis falls back to rules and narration to deterministic templates. The demo still runs end to end. |

Every annualized figure in the UI is rendered next to the assumption that
produced it.

---

## Status

Phase 0 — scaffold. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) §10 for the phase plan.
