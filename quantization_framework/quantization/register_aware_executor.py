import torch
try:
    from quantization.packing import ReQAPPackingPlanner
except ModuleNotFoundError:
    from .packing import ReQAPPackingPlanner

# Global context for MQF Register-Aware Execution
# Tracks scaling factors and carrying space across layers for hardware simulation
MQF_GLOBAL_CONTEXT = {
    'last_scale': 1.0,
    'is_first_layer': True,
    'register_size': 16,
    'carrying_bits': 0  # Number of bits reserved for carry/accumulation
}

def get_carrying_budget(packing_factor, w_bits, a_bits, register_size=16):
    """
    Calculates the carrying budget (remaining bits in the register 
    after segments are assigned) available for accumulation.
    """
    planner = ReQAPPackingPlanner(register_size=register_size, max_d=max(1, packing_factor))
    plan = planner.plan(w_bits, a_bits, d=packing_factor)
    return plan.lane_headroom_bits
