#  Demand tests
 
def test_demand_generator_is_seeded():
    from InventOps.demand import DemandGenerator
    gen1 = DemandGenerator("InventOps/data/demand_profiles.json", seed=42)
    gen2 = DemandGenerator("InventOps/data/demand_profiles.json", seed=42)
    samples1 = [gen1.sample("SKU_01", day) for day in range(10)]
    samples2 = [gen2.sample("SKU_01", day) for day in range(10)]
    assert samples1 == samples2, "DemandGenerator not deterministic for same seed"
 
 
def test_demand_generator_different_seeds_differ():
    from InventOps.demand import DemandGenerator
    gen1 = DemandGenerator("InventOps/data/demand_profiles.json", seed=1)
    gen2 = DemandGenerator("InventOps/data/demand_profiles.json", seed=2)
    samples1 = [gen1.sample("SKU_01", day) for day in range(20)]
    samples2 = [gen2.sample("SKU_01", day) for day in range(20)]
    assert samples1 != samples2, "Different seeds should produce different demand"
 