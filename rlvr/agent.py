from __future__ import annotations
import json
import os
from groq import Groq
from InventOps.models import Observation, Action


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_FAST_MODEL = "llama-3.1-8b-instant"   # for high-volume episode runs


class GroqAgent:
    """
    LLM-based agent using Groq inference.
    Converts Observation → structured prompt → Groq API → Action.
    """

    def __init__(self, system_prompt: str, model: str = GROQ_MODEL):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.system_prompt = system_prompt
        self.model = model

    def act(self, obs: Observation) -> tuple[Action, str]:
        """Returns (Action, raw_response_text)."""
        user_prompt = self._format_observation(obs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        action = self._parse_action(raw)
        return action, raw

    def _format_observation(self, obs: Observation) -> str:
        pending = "\n".join(
            f"  {o.sku_id}: {o.quantity} units → {o.warehouse_id} (day {o.arrival_day})"
            for o in obs.pending_orders
        ) or "  None"

        inventory_lines = []
        for sku, warehouses in obs.inventory_levels.items():
            wh_str = " | ".join(f"{wh}: {qty}" for wh, qty in warehouses.items())
            inventory_lines.append(f"  {sku}: {wh_str}")

        supplier_lines = "\n".join(
            f"  {sid}: {s.status.upper()}"
            + (f" (recovers day {s.recovery_day})" if s.status == "disrupted" else "")
            for sid, s in obs.supplier_status.items()
        )

        forecast_lines = "\n".join(
            f"  {sku}: {val:.1f} units/day"
            for sku, val in obs.demand_forecast.items()
        )

        budget_line = (
            f"\nBudget remaining: ${obs.budget_remaining:.0f}"
            if obs.budget_remaining is not None else ""
        )

        return f"""Day {obs.day} of {obs.episode_length}{budget_line}

INVENTORY:
{chr(10).join(inventory_lines)}

PENDING ORDERS (arriving):
{pending}

DEMAND FORECAST (7-day avg):
{forecast_lines}

WAREHOUSE CAPACITY REMAINING:
{chr(10).join(f'  {wh}: {cap} units' for wh, cap in obs.warehouse_capacity_remaining.items())}

SUPPLIER STATUS:
{supplier_lines}

Respond with a JSON object for ONE action:
{{"action_type": "order"|"transfer"|"hold", "sku_id": "...", "quantity": N, "target_warehouse": "...", "source_warehouse": "..."}}
"""

    @staticmethod
    def _parse_action(raw: str) -> Action:
        try:
            data = json.loads(raw)
            return Action(**data)
        except Exception:
            return Action(action_type="hold")
