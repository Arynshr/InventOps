from __future__ import annotations
import json
import os
from pathlib import Path
from groq import Groq
from InventOps import SupplyChainEnv
from rlvr.agent import GroqAgent


OPTIMIZER_MODEL = "llama-3.1-70b-versatile"
AGENT_MODEL     = "llama-3.1-8b-instant"


class PromptOptimizer:
    def __init__(
        self,
        task_id: str,
        initial_prompt_path: str,
        output_dir: str = "rlvr/prompts/rounds",
        episodes_per_round: int = 20,
        seeds: list[int] | None = None,
    ):
        self.task_id = task_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_round = episodes_per_round
        self.seeds = seeds or list(range(episodes_per_round))
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

        with open(initial_prompt_path) as f:
            self.current_prompt = f.read()

        self.history: list[dict] = []

    def run_round(self, round_num: int) -> dict:
        agent = GroqAgent(system_prompt=self.current_prompt, model=AGENT_MODEL)
        scores, failure_reasons = [], []

        for seed in self.seeds:
            env = SupplyChainEnv(task_id=self.task_id, seed=seed)
            obs = env.reset()
            done = False

            while not done:
                action, _ = agent.act(obs)
                obs, _, done, info = env.step(action)
                if info.get("failure_reason"):
                    failure_reasons.append(info["failure_reason"])

            scores.append(env.grade())

        mean_score = sum(scores) / len(scores)
        result = {
            "round": round_num,
            "mean_score": round(mean_score, 4),
            "min_score":  round(min(scores), 4),
            "max_score":  round(max(scores), 4),
            "failure_reasons": failure_reasons[:20],
        }
        self.history.append(result)

        prompt_path = self.output_dir / f"round_{round_num}.txt"
        prompt_path.write_text(self.current_prompt)

        print(f"Round {round_num}: mean={mean_score:.3f} "
              f"[{min(scores):.3f}, {max(scores):.3f}] "
              f"failures={len(failure_reasons)}")
        return result

    def refine_prompt(self, round_result: dict) -> str:
        failure_summary = "\n".join(
            f"- {r}" for r in set(round_result["failure_reasons"])
        ) or "None recorded"

        meta_prompt = f"""You are an expert prompt engineer for supply chain RL agents.

Current agent prompt achieved a mean score of {round_result['mean_score']:.3f} on the {self.task_id} task.

COMMON FAILURE REASONS OBSERVED:
{failure_summary}

CURRENT PROMPT:
---
{self.current_prompt}
---

Improve the prompt by adding 1-3 specific, actionable rules that address the failures above.
Do not remove existing rules. Keep the prompt under 600 words.
Return ONLY the updated prompt text, no preamble."""

        response = self.groq_client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()

    def run(self, num_rounds: int = 4) -> list[dict]:
        print(f"Starting RLVR: task={self.task_id}, "
              f"rounds={num_rounds}, episodes_per_round={self.episodes_per_round}")

        for r in range(1, num_rounds + 1):
            result = self.run_round(r)
            if r < num_rounds:
                self.current_prompt = self.refine_prompt(result)

        results_path = self.output_dir / "results.json"
        results_path.write_text(json.dumps(self.history, indent=2))
        print(f"\nResults saved to {results_path}")

        # Save final optimized prompt
        final_path = Path("rlvr/prompts/optimized_prompt.txt")
        final_path.write_text(self.current_prompt)
        print(f"Optimized prompt saved to {final_path}")

        return self.history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--rounds",   type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    optimizer = PromptOptimizer(
        task_id=args.task,
        initial_prompt_path="rlvr/prompts/base_prompt.txt",
        episodes_per_round=args.episodes,
    )
    history = optimizer.run(num_rounds=args.rounds)

    print("\n=== RLVR Score Progression ===")
    for h in history:
        bar = "█" * int(h["mean_score"] * 20)
        print(f"Round {h['round']}: {h['mean_score']:.3f} {bar}")
