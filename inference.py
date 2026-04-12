"""
inference.py — InventOps Submission Inference Script
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK   = "inventops"
TASKS       = ["easy", "medium", "hard"]
TEMPERATURE = 0.1  
MAX_TOKENS  = 60    


SYSTEM_PROMPT = (
    "Inventory manager. Respond with ONE JSON action only, no explanation.\n"
    "Order: {\"action_type\":\"order\",\"sku_id\":\"SKU_01\",\"quantity\":100,\"target_warehouse\":\"WH_1\"}\n"
    "Transfer: {\"action_type\":\"transfer\",\"sku_id\":\"SKU_01\",\"quantity\":50,\"source_warehouse\":\"WH_N\",\"target_warehouse\":\"WH_S\"}\n"
    "Hold: {\"action_type\":\"hold\"}\n"
    "Rules: never order from disrupted supplier. avoid stockouts. avoid capacity breach."
)


# ── Structured logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
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

def format_observation(obs) -> str:
    # Only show SKUs at risk (< 5 days cover)
    low_stock = []
    for sku, whs in obs.inventory_levels.items():
        total = sum(whs.values())
        forecast = obs.demand_forecast.get(sku, 0)
        days_cover = (total / forecast) if forecast > 0 else 99
        if days_cover < 5:
            wh_str = ",".join(f"{wh}:{qty}" for wh, qty in whs.items())
            low_stock.append(f"{sku}({wh_str},cover:{days_cover:.1f}d)")

    inv_str = " ".join(low_stock) if low_stock else "all_ok"

    # Pending orders — compact
    pending_str = " ".join(
        f"{o.sku_id}+{o.quantity}@{o.warehouse_id}d{o.arrival_day}"
        for o in obs.pending_orders
    ) or "none"

    # Disrupted suppliers only
    disrupted = [
        f"{sid}(rec:d{s.recovery_day})"
        for sid, s in obs.supplier_status.items()
        if s.status == "disrupted"
    ]
    sup_str = " ".join(disrupted) if disrupted else "all_ok"

    # Capacity free per warehouse
    cap_str = " ".join(
        f"{wh}:{free}free"
        for wh, free in obs.warehouse_capacity_remaining.items()
    )

    budget_str = f" budget:{obs.budget_remaining:.0f}" if obs.budget_remaining is not None else ""

    return (
        f"d{obs.day}/{obs.episode_length}{budget_str} "
        f"lowstock:{inv_str} "
        f"intransit:{pending_str} "
        f"disrupted:{sup_str} "
        f"capacity:{cap_str}"
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs):
    from InventOps.models import Action

    raw = '{"action_type":"hold"}'
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_observation(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model wraps JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        allowed = {"action_type", "sku_id", "quantity", "source_warehouse", "target_warehouse"}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return raw, Action(**filtered)

    except Exception as exc:
        print(f"[DEBUG] parse error: {exc} | raw={raw!r}", flush=True)
        return raw, Action(action_type="hold")


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> dict:
    from InventOps import SupplyChainEnv

    env = SupplyChainEnv(task_id=task_id, seed=42)
    obs = env.reset()

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

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
        print(f"[DEBUG] episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Guard: ensure we only run once even if container restarts
    if os.environ.get("INFERENCE_RUNNING"):
        print("[INFO] Already running, exiting.", flush=True)
        return
    os.environ["INFERENCE_RUNNING"] = "1"
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[INFO] model={MODEL_NAME} tasks={TASKS}", flush=True)

    results = []
    for i, task_id in enumerate(TASKS):
        result = run_episode(client, task_id)
        results.append(result)
        # Pause between tasks to avoid rate limit spikes
        if i < len(TASKS) - 1:
            time.sleep(3)

    print("\n[SUMMARY]", flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Steps':>7}", flush=True)
    print("-" * 28, flush=True)
    for r in results:
        print(f"{r['task_id']:<10} {r['score']:>7.3f} {r['steps']:>7}", flush=True)
    composite = sum(r["score"] for r in results) / len(results)
    print(f"\n{'Composite':<10} {composite:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
