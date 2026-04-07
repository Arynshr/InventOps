from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class PendingOrder(BaseModel):
    sku_id: str
    quantity: int
    warehouse_id: str
    arrival_day: int


class SupplierStatus(BaseModel):
    supplier_id: str
    status: Literal["available", "disrupted"]
    recovery_day: int | None = None


class Observation(BaseModel):
    day: int
    episode_length: int
    inventory_levels: dict[str, dict[str, int]]
    pending_orders: list[PendingOrder]
    demand_forecast: dict[str, float]
    warehouse_capacity_remaining: dict[str, int]
    supplier_status: dict[str, SupplierStatus]
    holding_costs: dict[str, float]
    stockout_penalties: dict[str, float]
    budget_remaining: float | None = None


class Action(BaseModel):
    action_type: Literal["order", "transfer", "hold"]
    sku_id: str | None = None
    quantity: int | None = None
    source_warehouse: str | None = None
    target_warehouse: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> Action:
        if self.action_type == "order":
            assert self.sku_id and self.quantity and self.target_warehouse, \
                "order requires sku_id, quantity, target_warehouse"
        if self.action_type == "transfer":
            assert self.sku_id and self.quantity \
                and self.source_warehouse and self.target_warehouse, \
                "transfer requires sku_id, quantity, source_warehouse, target_warehouse"
        return self


class StepReward(BaseModel):
    total: float
    fulfillment: float
    holding_cost: float
    stockout_penalty: float
    order_cost: float
    transfer_cost: float
    capacity_breach_penalty: float
    bullwhip_penalty: float


class EpisodeInfo(BaseModel):
    step: int
    action_valid: bool
    failure_reason: str | None = None
    units_sold: dict[str, int]
    stockout_units: dict[str, int]
    reward_breakdown: StepReward
