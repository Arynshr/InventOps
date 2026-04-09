"""
evaluate.py — Full benchmark evaluator for InventOps.
Runs hold-only, random, and (optionally) LLM agent across all 3 tasks.
Prints a formatted score table with mean ± std across N seeds.

Usage:
    python evaluate.py                             # hold + random agents only
    GROQ_API_KEY=gsk_... python evaluate.py --llm  # also runs Groq agent
    python evaluate.py --seeds 20 --llm
"""
from __future__ import annotations
import argparse
import os
import random
import statistics
from InventOps import SupplyChainEnv
from InventOps.models import Action


TASKS = ["easy", "medium", "hard"]


# ── Agents ────────────────────────────────────────────────────────────────────

def hold_agent(obs) -> Action:
    return Action(action_type="hold")


def random_agent(obs) -> Action:
    import random as _r
    choice = _r.choice(["hold", "hold", "order"])   # bias toward hold
    if choice == "order":
        skus = list(obs.inventory_levels.keys())
        warehouses = list(obs.warehouse_capacity_remaining.keys())
        sku = _r.choice(skus)
        wh  = _r.choice(warehouses)
        qty = _r.randint(10, 100)
        return Action(action_type="order", sku_id=sku, quantity=qty, target_warehouse=wh)
    return Action(action_type="hold")


def llm_agent_factory(prompt_path: str, model: str):
    from rlvr.agent import GroqAgent
    with open(prompt_path) as f:
        prompt = f.read()
    agent = GroqAgent(system_prompt=prompt, model=model)
    return lambda obs: agent.act(obs)[0]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_agent(agent_fn, task_id: str, seeds: list[int]) -> dict:
    scores = []
    for seed in seeds:
        env = SupplyChainEnv(task_id=task_id, seed=seed)
        obs = env.reset()
        done = False
        while not done:
            action = agent_fn(obs)
            obs, _, done, _ = env.step(action)
        scores.append(env.grade())
    return {
        "mean":   round(statistics.mean(scores), 4),
        "std":    round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        "min":    round(min(scores), 4),
        "max":    round(max(scores), 4),
        "n":      len(scores),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: dict, seeds: int):
    agents = list(results.keys())
    col_w = 22

    print(f"\n{'='*75}")
    print(f"  InventOps — Benchmark Evaluation  ({seeds} seeds per task)")
    print(f"{'='*75}")

    header = f"{'Task':<10}"
    for agent in agents:
        header += f"  {agent:^{col_w}}"
    print(header)

    sub = f"{'':10}"
    for _ in agents:
        sub += f"  {'mean ± std':^{col_w}}"
    print(sub)
    print(f"{'-'*75}")

    composites = {agent: [] for agent in agents}

    for task in TASKS:
        row = f"{task:<10}"
        for agent in agents:
            r = results[agent][task]
            composites[agent].append(r["mean"])
            cell = f"{r['mean']:.3f} ± {r['std']:.3f}"
            row += f"  {cell:^{col_w}}"
        print(row)

    print(f"{'-'*75}")
    composite_row = f"{'composite':<10}"
    for agent in agents:
        comp = round(sum(composites[agent]) / len(composites[agent]), 4)
        composite_row += f"  {comp:^{col_w}.3f}"
    print(composite_row)
    print(f"{'='*75}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InventOps full benchmark")
    parser.add_argument("--seeds",  type=int, default=20)
    parser.add_argument("--llm",    action="store_true", help="Include Groq LLM agent")
    parser.add_argument("--model",  default="llama-3.1-70b-versatile")
    parser.add_argument("--prompt", default="rlvr/prompts/base_prompt.txt")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    agents: dict[str, callable] = {
        "hold-only": hold_agent,
        "random":    random_agent,
    }

    if args.llm:
        if not os.environ.get("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY required for --llm flag")
        agents["groq-llm"] = llm_agent_factory(args.prompt, args.model)

        # Also run optimized prompt if it exists
        opt_path = "rlvr/prompts/optimized_prompt.txt"
        if os.path.exists(opt_path):
            agents["groq-optimized"] = llm_agent_factory(opt_path, args.model)

    results: dict[str, dict] = {agent: {} for agent in agents}

    for agent_name, agent_fn in agents.items():
        print(f"\nEvaluating: {agent_name}")
        for task_id in TASKS:
            print(f"  {task_id}...", end=" ", flush=True)
            results[agent_name][task_id] = run_agent(agent_fn, task_id, seeds)
            print(f"mean={results[agent_name][task_id]['mean']:.3f}")

    print_report(results, args.seeds)
