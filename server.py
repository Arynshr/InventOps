"""
server.py — FastAPI OpenEnv-compliant HTTP server for InventOps.
No openenv-core dependency — plain FastAPI only.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from InventOps import SupplyChainEnv
from InventOps.models import Action

app = FastAPI(title="InventOps OpenEnv Server")

# Single global env instance
_env: Optional[SupplyChainEnv] = None


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action: dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_dict(obs) -> dict:
    return {
        "day": obs.day,
        "episode_length": obs.episode_length,
        "inventory_levels": obs.inventory_levels,
        "pending_orders": [o.model_dump() for o in obs.pending_orders],
        "demand_forecast": obs.demand_forecast,
        "warehouse_capacity_remaining": obs.warehouse_capacity_remaining,
        "supplier_status": {k: v.model_dump() for k, v in obs.supplier_status.items()},
        "holding_costs": obs.holding_costs,
        "stockout_penalties": obs.stockout_penalties,
        "budget_remaining": obs.budget_remaining,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "inventops"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    global _env
    _env = SupplyChainEnv(task_id=request.task_id, seed=request.seed)
    obs = _env.reset()
    return {
        "observation": obs_to_dict(obs),
        "done": False,
    }


@app.post("/step")
def step(request: StepRequest):
    global _env
    if _env is None:
        _env = SupplyChainEnv(task_id="easy", seed=42)
        _env.reset()
    try:
        allowed = {"action_type", "sku_id", "quantity", "source_warehouse", "target_warehouse"}
        filtered = {k: v for k, v in request.action.items() if k in allowed}
        action = Action(**filtered)
    except Exception:
        action = Action(action_type="hold")

    obs, reward, done, info = _env.step(action)
    score = _env.grade() if done else None
    return {
        "observation": obs_to_dict(obs),
        "reward": reward,
        "done": done,
        "info": info,
        "score": score,
    }


@app.get("/state")
def state():
    if _env is None:
        return {"initialized": False}
    return _env.state()


@app.get("/schema")
def schema():
    return {
        "observation": {
            "type": "object",
            "properties": {
                "day": {"type": "integer"},
                "episode_length": {"type": "integer"},
                "inventory_levels": {"type": "object"},
                "pending_orders": {"type": "array"},
                "demand_forecast": {"type": "object"},
                "warehouse_capacity_remaining": {"type": "object"},
                "supplier_status": {"type": "object"},
                "budget_remaining": {"type": ["number", "null"]},
            }
        },
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["order", "transfer", "hold"]},
                "sku_id": {"type": "string"},
                "quantity": {"type": "integer"},
                "target_warehouse": {"type": "string"},
                "source_warehouse": {"type": "string"},
            },
            "required": ["action_type"]
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
