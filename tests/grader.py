import pytest
from InventOps import SupplyChainEnv
from InventOps.models import Action, Observation
 
 
# Grader tests 
 
@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_grader_returns_float_in_range(task_id):
    env = SupplyChainEnv(task_id=task_id, seed=42)
    obs = env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(Action(action_type="hold"))
    score = env.grade()
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
 
 
@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_grader_is_deterministic(task_id):
    """Same seed → same score across 3 independent runs."""
    scores = []
    for _ in range(3):
        env = SupplyChainEnv(task_id=task_id, seed=99)
        obs = env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(Action(action_type="hold"))
        scores.append(env.grade())
    assert scores[0] == scores[1] == scores[2], f"Non-deterministic scores: {scores}"
