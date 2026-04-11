"""
inference.py — InventOps Submission Inference Script
=====================================================
Runs all 3 tasks (easy, medium, hard) using an OpenAI-compatible LLM client.
Emits structured [START] / [STEP] / [END] logs to stdout.

Required environment variables:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key

Usage:
    HF_TOKEN=hf_... python inference.py
    HF_TOKEN=hf_... API_BASE_URL=https://... MODEL_NAME=... python inference.py
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

from openai import OpenAI

# Env config 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")

BENCHMARK    = "inventops"
TASKS        = ["easy", "medium", "hard"]
TEMPERATURE  = 0.2
MAX_TOKENS   = 256


# Structured logging 

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Sanitise action string: remove newlines so it stays on one line
    action_clean = action.replace("\n", " ").replace("\r", "")
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# System prompt 

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous inventory manager for a multi-warehouse supply chain.
    Your goal is to minimize total cost while maintaining high service level (fill demand).

    DECISION RULES:
    - Check supplier status before ordering. Disrupted suppliers cannot fulfil orders.
    - Check pending in-transit orders before placing new ones. Avoid duplicating orders.
    - Order early enough to cover lead time. If lead time is 3 days, order before you run out.
    - Transfer stock between warehouses when one has excess and another is low.
    - Hold if no action is needed. Never order speculatively.

    COST AWARENESS:
    - Holding cost accrues every day per unit held.
    - Stockout penalty = 2× unit margin per unmet unit. Stockouts are very expensive.
    - Fixed order cost charged per purchase order placed.
    - Capacity breach triggers a large fixed penalty. Never exceed warehouse capacity.

    Respond with EXACTLY ONE JSON object and nothing else:
    {"action_type": "order"|"transfer"|"hold", "sku_id": "...", "quantity": N,
     "target_warehouse": "...", "source_warehouse": "..."}

    For "hold": {"action_type": "hold"}
    For "order": {"action_type": "order", "sku_id": "SKU_01", "quantity": 150, "target_warehouse": "WH_1"}
    For "transfer": {"action_type": "transfer", "sku_id": "SKU_01", "quantity": 50,
                     "source_warehouse": "WH_N", "target_warehouse": "WH_S"}
""").strip()


# ── LLM call 

def format_observation(obs) -> str:
    """Convert Observation object to a compact LLM prompt."""
    inventory_lines = "\n".join(
        f"  {sku}: " + " | ".join(f"{wh}={qty}" for wh, qty in whs.items())
        for sku, whs in obs.inventory_levels.items()
    )
    pending = "\n".join(
        f"  {o.sku_id}: {o.quantity} units → {o.warehouse_id} (arrives day {o.arrival_day})"
        for o in obs.pending_orders
    ) or "  None"
    supplier_lines = "\n".join(
        f"  {sid}: {s.status.upper()}"
        + (f" (recovers day {s.recovery_day})" if s.status == "disrupted" else "")
        for sid, s in obs.supplier_status.items()
    )
    forecast_lines = "\n".join(
        f"  {sku}: {val:.1f} units/day" for sku, val in obs.demand_forecast.items()
    )
    budget_line = (
        f"\nBudget remaining: ${obs.budget_remaining:.0f}"
        if obs.budget_remaining is not None else ""
    )
    capacity_lines = "\n".join(
        f"  {wh}: {cap} units free" for wh, cap in obs.warehouse_capacity_remaining.items()
    )
    return (
        f"Day {obs.day} of {obs.episode_length}{budget_line}\n\n"
        f"INVENTORY:\n{inventory_lines}\n\n"
        f"IN TRANSIT:\n{pending}\n\n"
        f"DEMAND FORECAST (7-day avg):\n{forecast_lines}\n\n"
        f"WAREHOUSE CAPACITY FREE:\n{capacity_lines}\n\n"
        f"SUPPLIER STATUS:\n{supplier_lines}\n\n"
        f"Respond with ONE JSON action object."
    )


def get_action(client: OpenAI, obs) -> tuple[str, dict]:
    """Call LLM and parse response into an action dict. Falls back to hold."""
    from InventOps.models import Action

    user_prompt = format_observation(obs)
    raw = '{"action_type": "hold"}'
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model wraps JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        allowed = {"action_type", "sku_id", "quantity", "source_warehouse", "target_warehouse"}
        filtered = {k: v for k, v in data.items() if k in allowed}
        action = Action(**filtered)
        return raw, action

    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc} | raw={raw!r}", flush=True)
        return raw, Action(action_type="hold")


#  Episode runner 

def run_episode(client: OpenAI, task_id: str) -> dict:
    """Run one full episode for task_id. Returns result dict."""
    from InventOps import SupplyChainEnv

    env = SupplyChainEnv(task_id=task_id, seed=42)
    obs = env.reset()

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        step = 0
        while not done:
            step += 1
            raw, action = get_action(client, obs)
            obs, reward, done, info = env.step(action)

            error_msg = info.get("failure_reason")
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action.model_dump(exclude_none=True)),
                reward=reward,
                done=done,
                error=error_msg,
            )

        score = env.grade()
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# Main 

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] InventOps inference | model={MODEL_NAME} | tasks={TASKS}", flush=True)

    results = []
    for task_id in TASKS:
        result = run_episode(client, task_id)
        results.append(result)

    # Summary
    print("\n[SUMMARY]", flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Success':>9} {'Steps':>7}", flush=True)
    print("-" * 38, flush=True)
    for r in results:
        print(
            f"{r['task_id']:<10} {r['score']:>7.3f} {str(r['success']):>9} {r['steps']:>7}",
            flush=True,
        )
    composite = sum(r["score"] for r in results) / len(results)
    print(f"\n{'Composite':<10} {composite:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
