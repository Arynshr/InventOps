from InventOps.tasks.base import BaseTask


class EasyTask(BaseTask):
    config = {
        "episode_length": 30,
        "skus": ["SKU_01"],
        "initial_inventory": {"SKU_01": {"WH_1": 200}},
        "warehouse_capacity": {"WH_1": 1000},
        "warehouse_priority": ["WH_1"],
        "suppliers": [{"id": "SUP_A"}],
        "sku_supplier_map": {"SKU_01": "SUP_A"},
        "lead_time_distribution": {"type": "deterministic", "days": 3},
        "unit_margins": {"SKU_01": 5.0},
        "holding_costs": {"SKU_01": 0.02},
        "stockout_penalties": {"SKU_01": 2.0},
        "disruptions": {},
    }

    def grade(self, episode_log: list[dict]) -> float:
        total_days = len(episode_log)
        if total_days == 0:
            return 0.0

        stockout_days = sum(
            1 for step in episode_log
            if sum(step["info"].get("stockout_units", {}).values()) > 0
        )
        stockout_score = 1.0 - (stockout_days / total_days)

        total_reward = sum(step["reward"] for step in episode_log)
        oracle_reward = 120.0
        worst_reward = -500.0
        cost_score = max(0.0, (total_reward - worst_reward) / (oracle_reward - worst_reward))
        cost_score = min(1.0, cost_score)

        return round(0.50 * stockout_score + 0.50 * cost_score, 4)
