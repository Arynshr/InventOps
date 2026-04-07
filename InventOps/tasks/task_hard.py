from InventOps.tasks.base import BaseTask


class HardTask(BaseTask):
    config = {
        "episode_length": 90,
        "skus": [f"SKU_{i:02d}" for i in range(1, 26)],
        "initial_inventory": {
            f"SKU_{i:02d}": {"WH_N": 30, "WH_S": 30, "WH_C": 30}
            for i in range(1, 26)
        },
        "warehouse_capacity": {"WH_N": 8000, "WH_S": 8000, "WH_C": 8000},
        "warehouse_priority": ["WH_N", "WH_S", "WH_C"],
        "suppliers": [
            {"id": "SUP_A"}, {"id": "SUP_B"}, {"id": "SUP_C"}
        ],
        "sku_supplier_map": {
            **{f"SKU_{i:02d}": "SUP_A" for i in range(1, 10)},
            **{f"SKU_{i:02d}": "SUP_B" for i in range(10, 18)},
            **{f"SKU_{i:02d}": "SUP_C" for i in range(18, 26)},
        },
        "lead_time_distribution": {
            "type": "lognormal", "mu": 1.6, "sigma": 0.4,
            "min": 2, "max": 10
        },
        "unit_margins": {f"SKU_{i:02d}": float(i % 10 + 1) for i in range(1, 26)},
        "holding_costs": {f"SKU_{i:02d}": 0.02 for i in range(1, 26)},
        "stockout_penalties": {f"SKU_{i:02d}": 2.0 for i in range(1, 26)},
        "disruptions": {
            "probability_per_day": 0.033,
            "min_duration": 2,
            "max_duration": 5,
        },
    }

    def grade(self, episode_log: list[dict]) -> float:
        total_days = len(episode_log)
        if total_days == 0:
            return 0.0

        total_demand, total_fulfilled = 0, 0
        for step in episode_log:
            sold = step["info"].get("units_sold", {})
            stockout = step["info"].get("stockout_units", {})
            total_fulfilled += sum(sold.values())
            total_demand += sum(sold.values()) + sum(stockout.values())
        service_score = (total_fulfilled / total_demand) if total_demand > 0 else 0.0

        total_reward = sum(step["reward"] for step in episode_log)
        oracle_reward = 3500.0
        worst_reward = -15000.0
        cost_score = max(0.0, min(1.0,
            (total_reward - worst_reward) / (oracle_reward - worst_reward)
        ))

        transfer_steps = [
            step for step in episode_log
            if step["action"].get("action_type") == "transfer"
        ]
        if transfer_steps:
            useful = sum(
                1 for step in transfer_steps
                if sum(step["info"].get("stockout_units", {}).values()) == 0
            )
            transfer_score = useful / len(transfer_steps)
        else:
            transfer_score = 0.5

        disruption_steps = [
            i for i, step in enumerate(episode_log)
            if step["reward_breakdown"].get("stockout_penalty", 0) < -20
        ]
        if disruption_steps:
            recovery_penalties = []
            for d_step in disruption_steps:
                post = episode_log[d_step:min(d_step + 5, total_days)]
                recovery_penalties.append(
                    sum(s["reward_breakdown"].get("stockout_penalty", 0) for s in post)
                )
            disruption_score = max(0.0, 1.0 - abs(sum(recovery_penalties)) / 500.0)
        else:
            disruption_score = 1.0

        return round(
            0.35 * service_score +
            0.35 * cost_score +
            0.20 * transfer_score +
            0.10 * disruption_score,
            4
        )
