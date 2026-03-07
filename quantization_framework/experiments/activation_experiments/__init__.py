"""
Experimental Activation Quantization Module

This module provides tools for researching mixed-precision activation quantization:
- Per-layer activation sensitivity analysis
- Joint weight + activation bit-width search (with actual evaluation)
- Quantization-aware training with mixed-precision activations
- Ablation study runner comparing W+A configurations

WARNING: This is an experimental research module.
The main pipeline uses fixed 8-bit activations by default.

Version: 0.2.0 (experimental)
"""

__version__ = "0.2.0"
__status__ = "experimental"

# Module exports
from .activation_sensitivity import (
    measure_activation_sensitivity,
    ActivationSensitivityHook,
)
from .mixed_precision_search import (
    search_mixed_precision_config,
    calculate_bops,
    MixedPrecisionEvaluator,
)
from .validate_mixed_config import (
    validate_mixed_precision_config,
    MixedPrecisionValidator,
)
from .qat_mixed_precision import MixedPrecisionQATWrapper, train_mixed_qat
from .run_ablation_study import run_ablation_study

__all__ = [
    "measure_activation_sensitivity",
    "ActivationSensitivityHook",
    "search_mixed_precision_config",
    "calculate_bops",
    "MixedPrecisionEvaluator",
    "validate_mixed_precision_config",
    "MixedPrecisionValidator",
    "MixedPrecisionQATWrapper",
    "train_mixed_qat",
    "run_ablation_study",
]

print(f"[WARNING] Loaded experimental activation quantization module v{__version__}")
