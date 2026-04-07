from InventOps.tasks.base import BaseTask


class MediumTask(BaseTask):
    config = {
        "episode_length": 60,
        "skus": [f"SKU_{i:02d}" for i in range(1, 11)],
        "initial_inventory": {
            f"SKU_{i:02d}": {"WH_1": 50} for i in range(1, 11)
        },
        "warehouse_capacity": {"WH_1": 5000},
        "warehouse_priority": ["WH_1"],
        "suppliers": [{"id": "SUP_A"}, {"id": "SUP_B"}],
        "sku_supplier_map": {
            **{f"SKU_{i:02d}": "SUP_A" for i in range(1, 6)},
            **{f"SKU_{i:02d}": "SUP_B" for i in range(6, 11)},
        },
        "lead_time_distribution": {"type": "uniform", "min": 2, "max": 5},
        "unit_margins": {f"SKU_{i:02d}": float(i) for i in range(1, 11)},
        "holding_costs": {f"SKU_{i:02d}": 0.02 for i in range(1, 11)},
        "stockout_penalties": {f"SKU_{i:02d}": 2.0 for i in range(1, 11)},
        "daily_budget": 2500.0,
        "disruptions": {},
    }

    def grade(self, episode_log: list[dict]) -> float:
        total_days = len(episode_log)
        if total_days == 0:
            return 0.0

        total_demand, total_fulfilled = 0, 0
        for step in episode_log:
            sold = step["info"].get("units_sold", {})
            stockout = step["info"].get("stockout_units", {})
            fulfilled = sum(sold.values())
            unmet = sum(stockout.values())
            total_fulfilled += fulfilled
            total_demand += fulfilled + unmet

        fill_rate = total_fulfilled / total_demand if total_demand > 0 else 0.0
        fill_score = min(1.0, fill_rate / 0.95)

        total_reward = sum(step["reward"] for step in episode_log)
        oracle_reward = 800.0
        worst_reward = -3000.0
        cost_score = max(0.0, min(1.0,
            (total_reward - worst_reward) / (oracle_reward - worst_reward)
        ))

        breach_steps = sum(
            1 for step in episode_log
            if step["reward_breakdown"].get("capacity_breach_penalty", 0) < 0
        )
        capacity_score = 1.0 - min(1.0, breach_steps / total_days)

        return round(
            0.40 * fill_score +
            0.40 * cost_score +
            0.20 * capacity_score,
            4
        )
