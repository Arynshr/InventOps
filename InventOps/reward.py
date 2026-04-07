from InventOps.models import StepReward


HOLDING_RATE       = 0.02
STOCKOUT_MULT      = 2.0
FIXED_ORDER_COST   = 100.0
VAR_ORDER_COST     = 0.10
TRANSFER_COST      = 1.50
CAPACITY_PENALTY   = 50.0
BULLWHIP_THRESHOLD = 2.0
BULLWHIP_PENALTY   = 5.0


def compute_reward(
    inventory_held: dict[str, int],
    units_sold: dict[str, int],
    stockout_units: dict[str, int],
    unit_margins: dict[str, float],
    holding_costs: dict[str, float],
    orders_placed: int,
    units_ordered: int,
    units_transferred: int,
    capacity_breached: bool,
    order_variance: float,
    demand_variance: float,
) -> StepReward:

    fulfillment = sum(
        units_sold[sku] * unit_margins.get(sku, 1.0)
        for sku in units_sold
    )
    holding = -sum(
        inventory_held.get(sku, 0) * holding_costs.get(sku, HOLDING_RATE)
        for sku in inventory_held
    )
    stockout = -sum(stockout_units.values()) * STOCKOUT_MULT
    order_cost = -(orders_placed * FIXED_ORDER_COST + units_ordered * VAR_ORDER_COST)
    transfer = -units_transferred * TRANSFER_COST
    capacity = -CAPACITY_PENALTY if capacity_breached else 0.0

    bullwhip = 0.0
    if demand_variance > 0 and (order_variance / demand_variance) > BULLWHIP_THRESHOLD:
        bullwhip = -BULLWHIP_PENALTY * (order_variance / demand_variance)

    total = fulfillment + holding + stockout + order_cost + transfer + capacity + bullwhip

    return StepReward(
        total=round(total, 4),
        fulfillment=round(fulfillment, 4),
        holding_cost=round(holding, 4),
        stockout_penalty=round(stockout, 4),
        order_cost=round(order_cost, 4),
        transfer_cost=round(transfer, 4),
        capacity_breach_penalty=round(capacity, 4),
        bullwhip_penalty=round(bullwhip, 4),
    )
