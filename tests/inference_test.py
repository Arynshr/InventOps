"""
inference.py — InventOps Submission Inference Script
=====================================================
Runs all 3 tasks (easy, medium, hard) using an OpenAI-compatible LLM client.
Emits structured [START] / [STEP] / [END] logs to stdout.

Required environment variables:
    API_BASE_URL   LLM endpoint  (default: https://api.groq.com/openai/v1)
    MODEL_NAME     Model ID      (default: llama-3.1-70b-versatile)
    HF_TOKEN       API key       (your Groq / HF / OpenRouter key)

Usage:
    # Normal run
    HF_TOKEN=gsk_... python inference.py

    # Self-test (no API key needed — mocks LLM responses)
    python inference.py --test

    # Single task
    HF_TOKEN=gsk_... python inference.py --task easy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest.mock as mock
from io import StringIO
from typing import Optional

from openai import OpenAI

# ── Env config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-70b-versatile")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")

BENCHMARK    = "inventops"
TASKS        = ["easy", "medium", "hard"]
TEMPERATURE  = 0.2
MAX_TOKENS   = 256

# Score threshold above which an episode is considered "successful"
SUCCESS_THRESHOLD = 0.5


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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous inventory manager for a multi-warehouse supply chain.
Your goal is to minimize total cost while maintaining high service level (fulfill demand).

DECISION RULES:
- Check supplier status before ordering. Disrupted suppliers cannot fulfill orders.
- Check pending in-transit orders before placing new ones. Avoid duplicating orders.
- Order early enough to cover lead time. If lead time is 3 days, order before stockout.
- Transfer stock between warehouses when one has excess and another is low.
- Hold if no action is needed. Never order speculatively.

COST AWARENESS:
- Holding cost accrues every day per unit held.
- Stockout penalty = 2x unit margin per unmet unit. Stockouts are very expensive.
- Fixed order cost charged per purchase order placed.
- Capacity breach triggers a large fixed penalty. Never exceed warehouse capacity.

Respond with EXACTLY ONE JSON object and nothing else:
{"action_type": "order"|"transfer"|"hold", "sku_id": "...", "quantity": N,
 "target_warehouse": "...", "source_warehouse": "..."}

Examples:
  Hold:     {"action_type": "hold"}
  Order:    {"action_type": "order", "sku_id": "SKU_01", "quantity": 150, "target_warehouse": "WH_1"}
  Transfer: {"action_type": "transfer", "sku_id": "SKU_01", "quantity": 50, "source_warehouse": "WH_N", "target_warehouse": "WH_S"}"""


# ── Observation formatter ─────────────────────────────────────────────────────

def format_observation(obs) -> str:
    inventory_lines = "\n".join(
        f"  {sku}: " + " | ".join(f"{wh}={qty}" for wh, qty in whs.items())
        for sku, whs in obs.inventory_levels.items()
    )
    pending = "\n".join(
        f"  {o.sku_id}: {o.quantity} units -> {o.warehouse_id} (arrives day {o.arrival_day})"
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


# ── LLM call ──────────────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs):
    """Call LLM, parse JSON response into Action. Falls back to hold on any error."""
    from InventOps.models import Action

    raw = '{"action_type": "hold"}'
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

        # Strip markdown code fences if model wraps JSON
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
        print(f"[DEBUG] LLM parse error: {exc} | raw={raw!r}", flush=True)
        return raw, Action(action_type="hold")


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> dict:
    """Run one full episode. Returns result dict with score, steps, success."""
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
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error in {task_id}: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

class _InferenceTestSuite:
    """
    Built-in tests. Run with: python inference.py --test
    No API key required — LLM calls are mocked with hold actions.
    Tests:
      1. [START] / [STEP] / [END] format for all 3 tasks
      2. Field presence and naming
      3. Lowercase booleans (true/false not True/False)
      4. Score in [0.0, 1.0]
      5. Step count matches episode length
      6. rewards list length matches step count
      7. Fallback to hold on bad JSON from LLM
      8. Fallback to hold on LLM exception
    """

    EPISODE_LENGTHS = {"easy": 30, "medium": 60, "hard": 90}
    PASS = 0
    FAIL = 0

    def _green(self, s): return f"\033[32m{s}\033[0m"
    def _red(self, s):   return f"\033[31m{s}\033[0m"

    def ok(self, name):
        print(f"  {self._green('PASS')}  {name}")
        self.PASS += 1

    def fail(self, name, reason):
        print(f"  {self._red('FAIL')}  {name}: {reason}")
        self.FAIL += 1

    def assert_eq(self, name, got, expected):
        if got == expected:
            self.ok(name)
        else:
            self.fail(name, f"got {got!r}, expected {expected!r}")

    def assert_true(self, name, condition, reason="condition was False"):
        if condition:
            self.ok(name)
        else:
            self.fail(name, reason)

    def _mock_client(self, response_json: str = '{"action_type": "hold"}'):
        """Return a mocked OpenAI client that responds with response_json."""
        completion = mock.MagicMock()
        completion.choices[0].message.content = response_json
        client = mock.MagicMock()
        client.chat.completions.create.return_value = completion
        return client

    def _capture_episode(self, task_id: str, response_json: str = '{"action_type": "hold"}') -> tuple[str, dict]:
        """Run run_episode(), capture stdout, return (captured_output, result)."""
        client = self._mock_client(response_json)
        buf = StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            result = run_episode(client, task_id)
        finally:
            sys.stdout = real_stdout
        return buf.getvalue(), result

    # ── Individual test methods ───────────────────────────────────────────────

    def test_start_line_present(self, task_id: str, lines: list[str]):
        starts = [l for l in lines if l.startswith("[START]")]
        self.assert_eq(f"[{task_id}] exactly 1 [START] line", len(starts), 1)

    def test_end_line_present(self, task_id: str, lines: list[str]):
        ends = [l for l in lines if l.startswith("[END]")]
        self.assert_eq(f"[{task_id}] exactly 1 [END] line", len(ends), 1)

    def test_step_count(self, task_id: str, lines: list[str]):
        steps = [l for l in lines if l.startswith("[STEP]")]
        expected = self.EPISODE_LENGTHS[task_id]
        self.assert_eq(f"[{task_id}] step count = {expected}", len(steps), expected)

    def test_start_fields(self, task_id: str, lines: list[str]):
        start = next((l for l in lines if l.startswith("[START]")), "")
        for field in ["task=", "env=", "model="]:
            self.assert_true(
                f"[{task_id}] [START] has {field}",
                field in start,
                f"'{field}' not found in: {start}",
            )

    def test_step_fields(self, task_id: str, lines: list[str]):
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        if not step_lines:
            self.fail(f"[{task_id}] [STEP] fields", "no [STEP] lines found")
            return
        for field in ["step=", "action=", "reward=", "done=", "error="]:
            self.assert_true(
                f"[{task_id}] [STEP] has {field}",
                field in step_lines[0],
                f"'{field}' not found in: {step_lines[0]}",
            )

    def test_end_fields(self, task_id: str, lines: list[str]):
        end = next((l for l in lines if l.startswith("[END]")), "")
        for field in ["success=", "steps=", "score=", "rewards="]:
            self.assert_true(
                f"[{task_id}] [END] has {field}",
                field in end,
                f"'{field}' not found in: {end}",
            )

    def test_lowercase_booleans(self, task_id: str, lines: list[str]):
        all_lines = "\n".join(lines)
        self.assert_true(
            f"[{task_id}] booleans are lowercase",
            "done=True" not in all_lines and
            "done=False" not in all_lines and
            "success=True" not in all_lines and
            "success=False" not in all_lines,
            "Found uppercase True/False — must be lowercase true/false",
        )

    def test_score_in_range(self, task_id: str, result: dict):
        score = result["score"]
        self.assert_true(
            f"[{task_id}] score in [0.0, 1.0]",
            0.0 <= score <= 1.0,
            f"score={score} is out of range",
        )

    def test_rewards_length_matches_steps(self, task_id: str, result: dict):
        self.assert_eq(
            f"[{task_id}] len(rewards) == steps",
            len(result["rewards"]),
            result["steps"],
        )

    def test_end_rewards_count(self, task_id: str, lines: list[str], result: dict):
        end = next((l for l in lines if l.startswith("[END]")), "")
        try:
            rewards_part = end.split("rewards=")[1]
            count = len(rewards_part.split(","))
            self.assert_eq(
                f"[{task_id}] [END] rewards count matches steps",
                count,
                result["steps"],
            )
        except (IndexError, ValueError):
            self.fail(f"[{task_id}] [END] rewards count", "could not parse rewards= field")

    def test_fallback_on_bad_json(self):
        """LLM returns garbage JSON → should fall back to hold, not raise."""
        client = self._mock_client("this is not json at all !!!")
        buf = StringIO()
        sys.stdout = buf
        try:
            result = run_episode(client, "easy")
        except Exception as exc:
            sys.stdout = sys.__stdout__
            self.fail("fallback on bad JSON", f"raised exception: {exc}")
            return
        finally:
            sys.stdout = sys.__stdout__
        self.assert_true(
            "fallback on bad JSON returns score in [0,1]",
            0.0 <= result["score"] <= 1.0,
        )

    def test_fallback_on_llm_exception(self):
        """LLM raises an exception → should fall back to hold, not crash."""
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network timeout")
        buf = StringIO()
        sys.stdout = buf
        try:
            result = run_episode(client, "easy")
        except Exception as exc:
            sys.stdout = sys.__stdout__
            self.fail("fallback on LLM exception", f"raised: {exc}")
            return
        finally:
            sys.stdout = sys.__stdout__
        self.assert_true(
            "fallback on LLM exception returns score in [0,1]",
            0.0 <= result["score"] <= 1.0,
        )

    def test_markdown_fence_stripping(self):
        """LLM wraps JSON in ```json ... ``` — should still parse correctly."""
        fenced = '```json\n{"action_type": "hold"}\n```'
        client = self._mock_client(fenced)
        buf = StringIO()
        sys.stdout = buf
        try:
            result = run_episode(client, "easy")
        finally:
            sys.stdout = sys.__stdout__
        self.assert_true(
            "markdown fence stripping works",
            result["steps"] == 30,
            f"steps={result['steps']}, expected 30",
        )

    # ── Runner ────────────────────────────────────────────────────────────────

    def run(self):
        print("\n\033[1m━━━ inference.py self-test ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")

        for task_id in TASKS:
            print(f"\n  Task: {task_id}")
            output, result = self._capture_episode(task_id)
            lines = [l for l in output.splitlines() if l.startswith("[")]

            self.test_start_line_present(task_id, lines)
            self.test_end_line_present(task_id, lines)
            self.test_step_count(task_id, lines)
            self.test_start_fields(task_id, lines)
            self.test_step_fields(task_id, lines)
            self.test_end_fields(task_id, lines)
            self.test_lowercase_booleans(task_id, lines)
            self.test_score_in_range(task_id, result)
            self.test_rewards_length_matches_steps(task_id, result)
            self.test_end_rewards_count(task_id, lines, result)

        print("\n  Edge cases")
        self.test_fallback_on_bad_json()
        self.test_fallback_on_llm_exception()
        self.test_markdown_fence_stripping()

        total = self.PASS + self.FAIL
        print(f"\n\033[1m━━━ Results: {self.PASS}/{total} passed ━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")

        if self.FAIL == 0:
            print("\033[32m  ✓ All tests passed — inference.py is spec-compliant\033[0m\n")
            return True
        else:
            print(f"\033[31m  ✗ {self.FAIL} test(s) failed\033[0m\n")
            return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InventOps inference script")
    parser.add_argument("--test", action="store_true",
                        help="Run self-tests (no API key required)")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Run a single task instead of all three")
    args = parser.parse_args()

    if args.test:
        suite = _InferenceTestSuite()
        passed = suite.run()
        sys.exit(0 if passed else 1)

    # Normal inference run
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks_to_run = [args.task] if args.task else TASKS

    print(f"[INFO] InventOps inference | model={MODEL_NAME} | tasks={tasks_to_run}", flush=True)

    results = []
    for task_id in tasks_to_run:
        result = run_episode(client, task_id)
        results.append(result)

    # Summary table
    print("\n[SUMMARY]", flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Success':>9} {'Steps':>7}", flush=True)
    print("-" * 38, flush=True)
    for r in results:
        print(
            f"{r['task_id']:<10} {r['score']:>7.3f} "
            f"{str(r['success']):>9} {r['steps']:>7}",
            flush=True,
        )
    if len(results) > 1:
        composite = sum(r["score"] for r in results) / len(results)
        print(f"\n{'Composite':<10} {composite:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
