from __future__ import annotations
import copy
from InventOps.models import Observation, Action, PendingOrder, SupplierStatus
from InventOps.simulator import Simulator
from InventOps.demand import DemandGenerator
from InventOps.tasks.base import BaseTask


class SupplyChainEnv:
    """OpenEnv-compliant supply chain environment."""

    def __init__(self, task_id: str = "easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._task: BaseTask = self._load_task(task_id)
        self._demand_gen = DemandGenerator(
            profiles_path="InventOps/data/demand_profiles.json",
            seed=seed,
        )
        self._sim = Simulator(self._task.config, self._demand_gen)
        self._episode_log: list[dict] = []
        self._done = False

    def reset(self) -> Observation:
        self._demand_gen = DemandGenerator(
            profiles_path="InventOps/data/demand_profiles.json",
            seed=self.seed,
        )
        self._sim = Simulator(self._task.config, self._demand_gen)
        self._episode_log = []
        self._done = False
        return self._build_observation(self._sim._snapshot())

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        snapshot_before = self._sim._snapshot()
        snapshot_after, reward, info = self._sim.step(action)

        self._episode_log.append({
            "day": snapshot_before["day"],
            "action": action.model_dump(),
            "reward": reward.total,
            "reward_breakdown": reward.model_dump(),
            "info": info,
        })

        done = self._sim.day >= self._task.config["episode_length"]
        self._done = done

        obs = self._build_observation(snapshot_after)
        return obs, reward.total, done, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "day": self._sim.day,
            "done": self._done,
            "inventory": copy.deepcopy(self._sim.inventory),
            "pending_orders": [o.model_dump() for o in self._sim.pending_orders],
            "supplier_status": {
                k: v.model_dump() for k, v in self._sim.supplier_status.items()
            },
            "episode_log": self._episode_log,
        }

    def grade(self) -> float:
        if not self._done:
            raise RuntimeError("Episode not complete. Run to completion before grading.")
        return self._task.grade(self._episode_log)

    def _build_observation(self, snapshot: dict) -> Observation:
        forecasts = {
            sku: self._demand_gen.rolling_forecast(sku, snapshot["day"])
            for sku in self._task.config["skus"]
        }
        capacity_remaining = {}
        for wh_id, capacity in self._task.config["warehouse_capacity"].items():
            used = sum(
                snapshot["inventory"].get(sku, {}).get(wh_id, 0)
                for sku in self._task.config["skus"]
            )
            capacity_remaining[wh_id] = capacity - used

        return Observation(
            day=snapshot["day"],
            episode_length=self._task.config["episode_length"],
            inventory_levels=snapshot["inventory"],
            pending_orders=[PendingOrder(**o) for o in snapshot["pending_orders"]],
            demand_forecast=forecasts,
            warehouse_capacity_remaining=capacity_remaining,
            supplier_status={
                k: SupplierStatus(**v)
                for k, v in snapshot["supplier_status"].items()
            },
            holding_costs=self._task.config.get("holding_costs", {}),
            stockout_penalties=self._task.config.get("stockout_penalties", {}),
            budget_remaining=self._task.config.get("daily_budget"),
        )

    @staticmethod
    def _load_task(task_id: str) -> BaseTask:
        if task_id == "easy":
            from InventOps.tasks.task_easy import EasyTask
            return EasyTask()
        elif task_id == "medium":
            from InventOps.tasks.task_medium import MediumTask
            return MediumTask()
        elif task_id == "hard":
            from InventOps.tasks.task_hard import HardTask
            return HardTask()
        raise ValueError(f"Unknown task_id: {task_id}")
