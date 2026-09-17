"""Named plant-economics profiles.

A cost config is only meaningful for a specific asset. Pricing a milling tool with
turbofan economics produces numbers that are internally consistent and physically
absurd, so each dataset gets a profile built for the asset it actually describes,
with the reasoning written down next to the numbers.

These are starting points for the Cost Studio, not claims. Every field is editable
in the UI, and the what-if simulator exists precisely because the plant's own
numbers are the ones that matter.
"""

from __future__ import annotations

from kairos.decision.cost import CostConfig

#: Aero engine shop visit — the C-MAPSS asset. An unplanned in-service failure
#: means an AOG (aircraft on ground) event; planned work happens at a scheduled
#: maintenance slot. These are the figures used in the demo script.
TURBOFAN_PLANT = CostConfig(
    c_unplanned_repair=800_000,        # emergency module replacement + expedited parts
    c_planned_repair=120_000,          # scheduled overhaul slot
    c_inspection=15_000,               # borescope inspection, nothing found
    downtime_rate_per_hour=45_000,     # lost revenue while the aircraft is grounded
    unplanned_downtime_hours=12,
    planned_downtime_hours=3,
    c_secondary_damage=0.0,
    part_lead_time_cycles=5,           # flights between ordering and receiving a module
    value_per_remaining_cycle=2_000,   # amortised value of flight hours thrown away
    technician_capacity_per_day=3,
    cycles_per_year=500.0,             # ~500 flight cycles per engine per year
)

#: Milling machine / tooling — the AI4I asset. Failures are frequent and cheap
#: relative to an engine, which is exactly why the same model can deserve a
#: different operating point on each dataset.
MILLING_PLANT = CostConfig(
    c_unplanned_repair=25_000,         # broken tool, scrapped workpiece, emergency change
    c_planned_repair=6_000,            # scheduled tool change during a normal stop
    c_inspection=1_200,                # operator checks the tool, finds nothing
    downtime_rate_per_hour=9_000,      # lost throughput on the line
    unplanned_downtime_hours=3,
    planned_downtime_hours=0.5,
    c_secondary_damage=0.0,
    part_lead_time_cycles=0,           # tooling is held in stores
    value_per_remaining_cycle=0.0,     # a tool's remaining life has little resale value
    technician_capacity_per_day=6,
    observations_per_asset_year=10_000.0,   # the dataset's 10,000 cycles ~ one machine-year
)

PRESETS = {
    "turbofan": TURBOFAN_PLANT,
    "milling": MILLING_PLANT,
}


def describe(cfg: CostConfig) -> dict[str, float | str]:
    """The three numbers that explain any cost config at a glance."""
    return {
        "cost_of_a_miss": cfg.cost_unplanned_event,
        "cost_of_a_false_alarm": cfg.c_inspection,
        "consequence_ratio": cfg.consequence_ratio,
        "bayes_threshold": cfg.bayes_threshold,
        "summary": (
            f"One missed failure costs {cfg.currency} {cfg.cost_unplanned_event:,.0f}; one "
            f"false alarm costs {cfg.currency} {cfg.c_inspection:,.0f}. That ratio of "
            f"{cfg.consequence_ratio:,.0f}:1 puts the break-even probability at "
            f"{cfg.bayes_threshold:.1%} — alert on anything more likely than that."
        ),
    }
