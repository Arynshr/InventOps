from __future__ import annotations
import json
import numpy as np
from pathlib import Path


class DemandGenerator:
    """
    Generates stochastic daily demand per SKU.
    Distributions calibrated from M5 Forecasting dataset.
    """

    def __init__(self, profiles_path: str, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        with open(profiles_path) as f:
            self.profiles: dict = json.load(f)

    def sample(self, sku_id: str, day: int) -> int:
        p = self.profiles[sku_id]
        seasonal_factor = p["seasonality"][day % 7]
        trend_factor = 1.0 + p["trend"] * day
        mu = p["mean"] * seasonal_factor * trend_factor

        if p["distribution"] == "normal":
            raw = self.rng.normal(mu, p["std"])
        elif p["distribution"] == "poisson":
            raw = self.rng.poisson(mu)
        else:
            raise ValueError(f"Unknown distribution: {p['distribution']}")

        return max(0, int(raw))

    def rolling_forecast(self, sku_id: str, day: int, window: int = 7) -> float:
        """7-day rolling average demand forecast."""
        return float(np.mean([
            self.profiles[sku_id]["mean"]
            * self.profiles[sku_id]["seasonality"][(day + i) % 7]
            for i in range(window)
        ]))
