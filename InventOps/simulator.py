from __future__ import annotations
import copy
import numpy as np
from InventOps.models import Action, PendingOrder, SupplierStatus
from InventOps.demand import DemandGenerator
from InventOps.reward import compute_reward, StepReward


class Simulator:
    def __init__(self, task_config: dict, demand_gen: DemandGenerator):
        self.cfg = task_config
        self.demand_gen = demand_gen
        self._reset_state()

    def _reset_state(self):
        self.day = 0
        self.inventory: dict[str, dict[str, int]] = copy.deepcopy(
            self.cfg["initial_inventory"]
        )
        self.pending_orders: list[PendingOrder] = []
        self.supplier_status: dict[str, SupplierStatus] = {
            s["id"]: SupplierStatus(supplier_id=s["id"], status="available")
            for s in self.cfg["suppliers"]
        }
        self.warehouse_capacity: dict[str, int] = copy.deepcopy(
            self.cfg["warehouse_capacity"]
        )
        self._order_history: list[int] = []
        self._demand_history: list[int] = []

    def step(self, action: Action) -> tuple[dict, StepReward, dict]:
        info: dict = {"action_valid": True, "failure_reason": None}

        valid, reason = self._validate(action)
        if not valid:
            info["action_valid"] = False
            info["failure_reason"] = reason
            action = Action(action_type="hold")

        orders_placed, units_ordered, units_transferred = 0, 0, 0

        if action.action_type == "order":
            lead_time = self._sample_lead_time(action.sku_id)
            self.pending_orders.append(PendingOrder(
                sku_id=action.sku_id,
                quantity=action.quantity,
                warehouse_id=action.target_warehouse,
                arrival_day=self.day + lead_time,
            ))
            orders_placed = 1
            units_ordered = action.quantity
            self._order_history.append(action.quantity)

        elif action.action_type == "transfer":
            transferred = min(
                action.quantity,
                self.inventory[action.sku_id].get(action.source_warehouse, 0)
            )
            self.inventory[action.sku_id][action.source_warehouse] -= transferred
            self.inventory[action.sku_id][action.target_warehouse] = (
                self.inventory[action.sku_id].get(action.target_warehouse, 0) + transferred
            )
            units_transferred = transferred

        arriving = [o for o in self.pending_orders if o.arrival_day == self.day]
        for order in arriving:
            self.inventory[order.sku_id][order.warehouse_id] = (
                self.inventory[order.sku_id].get(order.warehouse_id, 0) + order.quantity
            )
        self.pending_orders = [o for o in self.pending_orders if o.arrival_day != self.day]

        units_sold: dict[str, int] = {}
        stockout_units: dict[str, int] = {}
        total_demand: int = 0

        for sku_id in self.cfg["skus"]:
            demand = self.demand_gen.sample(sku_id, self.day)
            total_demand += demand
            self._demand_history.append(demand)

            remaining = demand
            sold = 0
            for wh in self.cfg.get("warehouse_priority", list(self.inventory[sku_id])):
                available = self.inventory[sku_id].get(wh, 0)
                fulfill = min(available, remaining)
                self.inventory[sku_id][wh] -= fulfill
                sold += fulfill
                remaining -= fulfill
                if remaining == 0:
                    break

            units_sold[sku_id] = sold
            stockout_units[sku_id] = max(0, demand - sold)

        capacity_breached = False
        for wh_id, capacity in self.warehouse_capacity.items():
            total_in_wh = sum(
                self.inventory[sku].get(wh_id, 0)
                for sku in self.cfg["skus"]
            )
            if total_in_wh > capacity:
                capacity_breached = True

        inventory_held = {
            sku: sum(self.inventory[sku].values())
            for sku in self.cfg["skus"]
        }
        order_var = float(np.var(self._order_history[-10:])) if self._order_history else 0.0
        demand_var = float(np.var(self._demand_history[-10:])) if self._demand_history else 1.0

        reward = compute_reward(
            inventory_held=inventory_held,
            units_sold=units_sold,
            stockout_units=stockout_units,
            unit_margins=self.cfg.get("unit_margins", {}),
            holding_costs=self.cfg.get("holding_costs", {}),
            orders_placed=orders_placed,
            units_ordered=units_ordered,
            units_transferred=units_transferred,
            capacity_breached=capacity_breached,
            order_variance=order_var,
            demand_variance=demand_var,
        )

        self._tick_supplier_disruptions()
        self.day += 1

        info["units_sold"] = units_sold
        info["stockout_units"] = stockout_units
        info["reward_breakdown"] = reward

        return self._snapshot(), reward, info

    def _validate(self, action: Action) -> tuple[bool, str | None]:
        if action.action_type == "hold":
            return True, None
        if action.action_type == "order":
            supplier = self._sku_supplier(action.sku_id)
            if supplier and self.supplier_status[supplier].status == "disrupted":
                return False, f"Supplier {supplier} is disrupted"
            if action.quantity <= 0:
                return False, "Order quantity must be positive"
        if action.action_type == "transfer":
            available = self.inventory.get(action.sku_id, {}).get(action.source_warehouse, 0)
            if available < action.quantity:
                return False, f"Insufficient stock: {available} < {action.quantity}"
        return True, None

    def _sample_lead_time(self, sku_id: str) -> int:
        dist = self.cfg.get("lead_time_distribution", {"type": "deterministic", "days": 3})
        if dist["type"] == "deterministic":
            return dist["days"]
        elif dist["type"] == "uniform":
            return int(self.demand_gen.rng.integers(dist["min"], dist["max"] + 1))
        elif dist["type"] == "lognormal":
            raw = self.demand_gen.rng.lognormal(dist["mu"], dist["sigma"])
            return int(np.clip(raw, dist.get("min", 1), dist.get("max", 14)))
        raise ValueError(f"Unknown lead time distribution: {dist['type']}")

    def _tick_supplier_disruptions(self):
        disruption_cfg = self.cfg.get("disruptions", {})
        prob = disruption_cfg.get("probability_per_day", 0.0)
        for supplier_id, status in self.supplier_status.items():
            if status.status == "available":
                if self.demand_gen.rng.random() < prob:
                    duration = int(self.demand_gen.rng.integers(
                        disruption_cfg.get("min_duration", 2),
                        disruption_cfg.get("max_duration", 5) + 1
                    ))
                    self.supplier_status[supplier_id] = SupplierStatus(
                        supplier_id=supplier_id,
                        status="disrupted",
                        recovery_day=self.day + duration,
                    )
            elif status.recovery_day and self.day >= status.recovery_day:
                self.supplier_status[supplier_id] = SupplierStatus(
                    supplier_id=supplier_id, status="available"
                )

    def _sku_supplier(self, sku_id: str) -> str | None:
        return self.cfg.get("sku_supplier_map", {}).get(sku_id)

    def _snapshot(self) -> dict:
        return {
            "day": self.day,
            "inventory": copy.deepcopy(self.inventory),
            "pending_orders": [o.model_dump() for o in self.pending_orders],
            "supplier_status": {
                k: v.model_dump() for k, v in self.supplier_status.items()
            },
            "warehouse_capacity": copy.deepcopy(self.warehouse_capacity),
        }
