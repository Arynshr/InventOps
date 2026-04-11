# Reward tests 
 
def test_reward_components_sum_to_total():
    from InventOps.reward import compute_reward
    reward = compute_reward(
        inventory_held={"SKU_01": 100},
        units_sold={"SKU_01": 50},
        stockout_units={"SKU_01": 0},
        unit_margins={"SKU_01": 5.0},
        holding_costs={"SKU_01": 0.02},
        orders_placed=1,
        units_ordered=100,
        units_transferred=0,
        capacity_breached=False,
        order_variance=0.0,
        demand_variance=1.0,
    )
    computed_total = round(
        reward.fulfillment + reward.holding_cost + reward.stockout_penalty +
        reward.order_cost + reward.transfer_cost +
        reward.capacity_breach_penalty + reward.bullwhip_penalty,
        4
    )
    assert abs(reward.total - computed_total) < 0.001, \
        f"Total mismatch: {reward.total} vs {computed_total}"
 
 
def test_stockout_increases_penalty():
    from InventOps.reward import compute_reward
    base = compute_reward(
        inventory_held={"SKU_01": 0}, units_sold={"SKU_01": 50},
        stockout_units={"SKU_01": 0}, unit_margins={"SKU_01": 5.0},
        holding_costs={"SKU_01": 0.02}, orders_placed=0, units_ordered=0,
        units_transferred=0, capacity_breached=False,
        order_variance=0.0, demand_variance=1.0,
    )
    with_stockout = compute_reward(
        inventory_held={"SKU_01": 0}, units_sold={"SKU_01": 30},
        stockout_units={"SKU_01": 20}, unit_margins={"SKU_01": 5.0},
        holding_costs={"SKU_01": 0.02}, orders_placed=0, units_ordered=0,
        units_transferred=0, capacity_breached=False,
        order_variance=0.0, demand_variance=1.0,
    )
    assert with_stockout.total < base.total
 