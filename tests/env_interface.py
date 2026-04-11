import pytest
from InventOps import SupplyChainEnv
from InventOps.models import Action, Observation
 
# Env interface tests 

def test_reset_produces_clean_state():
    env = SupplyChainEnv("easy", seed=42)
    obs1 = env.reset()
 
    for _ in range(5):
        env.step(Action(action_type="hold"))
 
    obs2 = env.reset()
    assert obs1.day == obs2.day == 0
    assert obs1.inventory_levels == obs2.inventory_levels
 
 
def test_step_raises_after_done():
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(Action(action_type="hold"))
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(Action(action_type="hold"))
 
 
def test_step_returns_correct_types():
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="hold"))
    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
 
 
def test_grade_raises_before_done():
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    with pytest.raises(RuntimeError, match="Episode not complete"):
        env.grade()
 
 
def test_state_returns_correct_keys():
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    state = env.state()
    for key in ["task_id", "seed", "day", "done", "inventory", "episode_log"]:
        assert key in state, f"Missing key in state(): {key}"
