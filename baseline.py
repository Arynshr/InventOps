"""
Baseline inference script.
Uses Groq API to run an LLM agent on all 3 tasks.
Reads credentials from GROQ_API_KEY environment variable.
Produces reproducible scores: same seeds → same scores.

Usage:
    GROQ_API_KEY=gsk_... python baseline.py
    GROQ_API_KEY=gsk_... python baseline.py --model llama-3.1-70b-versatile --seeds 10
    GROQ_API_KEY=gsk_... python baseline.py --prompt rlvr/prompts/optimized_prompt.txt
"""
import argparse
import os
import statistics
from InventOps import SupplyChainEnv
from rlvr.agent import GroqAgent


TASKS         = ["easy", "medium", "hard"]
DEFAULT_MODEL = "llama-3.1-70b-versatile"
DEFAULT_PROMPT = "rlvr/prompts/base_prompt.txt"


def run_baseline(model: str, seeds: list[int], prompt_path: str) -> dict:
    with open(prompt_path) as f:
        prompt = f.read()

    agent = GroqAgent(system_prompt=prompt, model=model)
    results = {}

    for task_id in TASKS:
        scores = []
        print(f"\nRunning task: {task_id} ({len(seeds)} seeds)...")
        for seed in seeds:
            env = SupplyChainEnv(task_id=task_id, seed=seed)
            obs = env.reset()
            done = False
            while not done:
                action, _ = agent.act(obs)
                obs, _, done, _ = env.step(action)
            scores.append(env.grade())
            print(f"  seed={seed:3d}  score={scores[-1]:.4f}", flush=True)

        results[task_id] = {
            "mean":   round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "std":    round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
            "min":    round(min(scores), 4),
            "max":    round(max(scores), 4),
            "n":      len(scores),
        }

    composite = round(sum(results[t]["mean"] for t in TASKS) / len(TASKS), 4)
    results["composite"] = composite
    return results


def print_results(results: dict, model: str):
    n = results.get("easy", {}).get("n", "?")
    print(f"\n{'='*60}")
    print("  InventOps — Baseline Evaluation")
    print(f"  Model : {model}")
    print(f"  Seeds : {n} per task")
    print(f"{'='*60}")
    print(f"{'Task':<12} {'Score':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"{'-'*60}")
    for task in TASKS:
        r = results[task]
        print(f"{task:<12} {r['mean']:>7.3f} {r['std']:>7.3f} "
              f"{r['min']:>7.3f} {r['max']:>7.3f}")
    print(f"{'-'*60}")
    print(f"{'Composite':<12} {results['composite']:>7.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    parser = argparse.ArgumentParser(description="InventOps baseline evaluation")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--seeds",  type=int, default=50,
                        help="Number of seeds (episodes) per task")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results = run_baseline(args.model, seeds, args.prompt)
    print_results(results, args.model)
