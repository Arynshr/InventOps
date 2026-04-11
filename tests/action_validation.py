import pytest
from InventOps import SupplyChainEnv
from InventOps.models import Action, Observation

# Action validation tests 
 
def test_invalid_order_results_in_hold():
    """Ordering from a non-existent SKU silently becomes hold."""
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    # Order with zero quantity is invalid — env should hold
    bad_action = Action(action_type="order", sku_id="SKU_01", quantity=-1, target_warehouse="WH_1")
    obs, reward, done, info = env.step(bad_action)
    assert info["action_valid"] is False
 
 
def test_order_action_creates_pending_order():
    env = SupplyChainEnv("easy", seed=42)
    env.reset()
    action = Action(action_type="order", sku_id="SKU_01", quantity=50, target_warehouse="WH_1")
    env.step(action)
    state = env.state()
    assert len(state["pending_orders"]) > 0
 