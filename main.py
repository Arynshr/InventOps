"""
main.py — Quick smoke test. Runs all 3 tasks with a hold-only agent.
Use baseline.py for the full Groq-backed evaluation.
"""
from InventOps import SupplyChainEnv
from InventOps.models import Action


def main():
    print("InventOps — Smoke Test (hold-only agent)\n")
    for task_id in ["easy", "medium", "hard"]:
        env = SupplyChainEnv(task_id=task_id, seed=42)
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = Action(action_type="hold")
            obs, reward, done, info = env.step(action)
            total_reward += reward
        score = env.grade()
        print(f"  {task_id:<8}  score={score:.4f}  total_reward={total_reward:.1f}")
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
