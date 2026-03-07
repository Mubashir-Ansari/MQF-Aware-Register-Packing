import torch
import math
try:
    from quantization.packing import ReQAPPackingPlanner
except ModuleNotFoundError:
    from .packing import ReQAPPackingPlanner

class RegisterPackingSimulator:
    """
    Simulates FPGA register packing for mixed-precision MAC operations.
    Formula: d * (2^W - 1) * (2^A - 1) < 2^(R/d)
    
    Where:
    - R: Total Register Size (e.g. 16-bit)
    - d: Packing Factor (number of weights/activations in one register)
    - W: Weight bit-width
    - A: Activation bit-width
    """
    def __init__(self, register_size=16):
        self.R = register_size
        self._planner = ReQAPPackingPlanner(register_size=register_size, max_d=8)

    def is_valid_packing(self, w_bits, a_bits, d):
        """Check if (W, A, d) triplet fits in the register without overflow."""
        return self._planner.is_safe(w_bits, a_bits, d)

    def find_max_packing_factor(self, w_bits, a_bits, max_d=8):
        """
        Find highest valid packing factor d under Eq.1.

        By default we cap d at 8 (practical SIMD lane cap), but we now
        evaluate all integer values in [1, max_d] instead of only powers of two.
        """
        self._planner.max_d = int(max_d)
        return self._planner.best_factor(w_bits, a_bits)

    def get_carrying_budget(self, w_bits, a_bits, d):
        """
        Calculates how many 'extra' bits are available in the segment
        to handle accumulation carries without overflow.
        """
        if d <= 0: return 0
        segment_size = self.R // d
        
        # Max value of a single multiplication
        max_prod = (2**w_bits - 1) * (2**a_bits - 1)
        if max_prod == 0: return segment_size
        
        # Bits needed for product
        prod_bits = math.ceil(math.log2(max_prod + 1))
        
        # Residual bits for carrying (accumulation)
        carrying_bits = segment_size - prod_bits
        return max(0, carrying_bits)

    def get_packing_efficiency(self, w_bits, a_bits):
        """Returns the throughput gain (d), utilization %, and carrying budget."""
        d = self.find_max_packing_factor(w_bits, a_bits)
        segment_size = self.R // d
        
        bits_used = w_bits + a_bits
        utilization = (bits_used / segment_size) * 100
        carry_budget = self.get_carrying_budget(w_bits, a_bits, d)
        
        return d, utilization, carry_budget

    def calculate_register_savings(self, layer_config, num_params):
        """
        Calculates total bits saved vs 32-bit float and 8-bit baseline.
        Returns (bits_saved_vs_fp32, bits_saved_vs_int8)
        """
        w_bits = layer_config.get('weight', 8)
        
        total_bits_fp32 = num_params * 32
        total_bits_int8 = num_params * 8
        total_bits_hrp = num_params * w_bits
        
        return (total_bits_fp32 - total_bits_hrp), (total_bits_int8 - total_bits_hrp)

def calculate_model_throughput(model_config, register_size=16):
    """
    Calculate total throughput gain across all layers.
    Throughput = Sum(MACs_i * d_i) / Sum(MACs_i)
    """
    sim = RegisterPackingSimulator(register_size)
    total_weighted_d = 0
    total_macs = 0
    
    # This requires layer MAC info which we'll get from the engine
    # For now, this is a skeleton for the search engine to use
    pass
