import torch

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
    segment_size = register_size // packing_factor
    payload_bits = w_bits + a_bits
    
    # Simple segment-based carry budget
    # The segment must hold (2^W - 1) * (2^A - 1) * (some_accumulation_factor)
    return segment_size - payload_bits
