"""Request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kairos.decision.cost import CostConfig


class CostConfigIn(BaseModel):
    """Plant economics from the Cost Studio. Mirrors CostConfig, validated."""

    c_unplanned_repair: float = Field(ge=0)
    c_planned_repair: float = Field(ge=0)
    c_inspection: float = Field(ge=0)
    downtime_rate_per_hour: float = Field(ge=0)
    unplanned_downtime_hours: float = Field(ge=0)
    planned_downtime_hours: float = Field(ge=0)
    c_secondary_damage: float = Field(default=0.0, ge=0)
    part_lead_time_cycles: int = Field(default=0, ge=0)
    value_per_remaining_cycle: float = Field(default=0.0, ge=0)
    technician_capacity_per_day: int = Field(default=3, ge=0)
    currency: str = "INR"
    cycles_per_year: float = Field(default=500.0, gt=0)
    observations_per_asset_year: float = Field(default=10_000.0, gt=0)

    def to_config(self) -> CostConfig:
        return CostConfig(**self.model_dump())

    @classmethod
    def from_config(cls, cfg: CostConfig) -> CostConfigIn:
        return cls(**{k: v for k, v in cfg.__dict__.items()})


class DatasetSelect(BaseModel):
    builtin: Literal["cmapss", "ai4i"] | None = None
    name: str | None = None


class RunCreate(BaseModel):
    dataset_id: str
    cost_config: CostConfigIn | None = None
    time_budget_s: float = Field(default=120.0, gt=0, le=1800)


class ApprovalIn(BaseModel):
    approved: bool = True
    plan: dict[str, Any] | None = None
    note: str | None = None


class SimulateIn(BaseModel):
    cost_config: CostConfigIn
