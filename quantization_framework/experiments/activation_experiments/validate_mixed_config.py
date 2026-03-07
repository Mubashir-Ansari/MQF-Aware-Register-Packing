"""
Validate Mixed-Precision Configuration (Weight + Activation)

Applies both weight AND activation quantization per configuration,
calibrates per-layer activation quantizers, and evaluates final accuracy.

Usage:
    python validate_mixed_config.py \
        --model vgg11_bn \
        --checkpoint path/to/checkpoint.pth \
        --config mixed_config.json \
        --dataset cifar10 \
        --output validation_results.json
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)


class CalibrationActivationQuantizer(nn.Module):
    """
    Activation quantizer with calibration support.
    Collects min/max statistics during calibration, then applies fixed quantization.
    """
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.calibrating = False
        self.calibrated = False

        # Statistics collected during calibration
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def calibrate_mode(self, enable=True):
        """Enable/disable calibration mode."""
        self.calibrating = enable
        if not enable and self.min_val < float('inf'):
            # Finalize calibration
            self._compute_quantization_params()
            self.calibrated = True

    def _compute_quantization_params(self):
        """Compute scale and zero point from collected statistics."""
        q_min = 0
        q_max = 2 ** self.bit_width - 1

        # Ensure buffers are on same device
        min_val = self.min_val.item()
        max_val = self.max_val.item()

        scale = (max_val - min_val) / (q_max - q_min)
        scale = max(scale, 1e-8)  # Avoid division by zero

        zero_point = round(-min_val / scale)
        zero_point = max(q_min, min(q_max, zero_point))

        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)

    def forward(self, x):
        if self.calibrating:
            # Collect statistics
            with torch.no_grad():
                batch_min = x.min()
                batch_max = x.max()

                # Move buffers to same device if needed
                if self.min_val.device != x.device:
                    self.min_val = self.min_val.to(x.device)
                    self.max_val = self.max_val.to(x.device)
                    self.scale = self.scale.to(x.device)
                    self.zero_point = self.zero_point.to(x.device)

                self.min_val = torch.min(self.min_val, batch_min)
                self.max_val = torch.max(self.max_val, batch_max)

            return x  # Pass through during calibration

        if not self.calibrated:
            return x  # Not calibrated, pass through

        # Apply quantization
        q_min = 0
        q_max = 2 ** self.bit_width - 1

        # Move buffers if needed
        if self.scale.device != x.device:
            self.scale = self.scale.to(x.device)
            self.zero_point = self.zero_point.to(x.device)

        x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
        x_dq = (x_int - self.zero_point) * self.scale

        return x_dq


class MixedPrecisionValidator:
    """
    Applies mixed-precision quantization for both weights and activations.
    """
    def __init__(self, model, config):
        """
        Args:
            model: PyTorch model
            config: Mixed precision config
                    Format: {layer_name: {"weight": bits, "activation": bits}}
        """
        self.model = model
        self.config = config
        self.activation_quantizers = {}
        self.hooks = []

    def _get_activation_config(self):
        """Extract activation bit-widths from config."""
        act_config = {}
        for layer_name, cfg in self.config.items():
            if isinstance(cfg, dict) and 'activation' in cfg:
                act_config[layer_name] = cfg['activation']
        return act_config

    def _get_weight_config(self):
        """Extract weight bit-widths from config."""
        w_config = {}
        for layer_name, cfg in self.config.items():
            if isinstance(cfg, dict) and 'weight' in cfg:
                w_config[layer_name] = cfg['weight']
            elif isinstance(cfg, int):
                # Legacy format: just weight bits
                w_config[layer_name] = cfg
        return w_config

    def apply_weight_quantization(self):
        """Apply weight quantization according to config."""
        w_config = self._get_weight_config()

        for name, module in self.model.named_modules():
            if name in w_config and hasattr(module, 'weight') and module.weight is not None:
                bits = w_config[name]
                w = module.weight.data
                q_w, _, _ = quantize_tensor(w, bit_width=bits)
                module.weight.data = q_w

    def setup_activation_quantizers(self):
        """Install activation quantizers as forward hooks."""
        act_config = self._get_activation_config()

        for name, module in self.model.named_modules():
            if name in act_config:
                bits = act_config[name]
                quantizer = CalibrationActivationQuantizer(bit_width=bits)
                self.activation_quantizers[name] = quantizer

                # Create hook
                def make_hook(quant):
                    def hook(mod, inp, out):
                        return quant(out)
                    return hook

                handle = module.register_forward_hook(make_hook(quantizer))
                self.hooks.append(handle)

    def calibrate(self, dataloader, device='cuda', num_batches=50):
        """
        Calibrate activation quantizers using representative data.

        Args:
            dataloader: Data loader for calibration
            device: Device to use
            num_batches: Number of batches for calibration
        """
        # Enable calibration mode
        for quantizer in self.activation_quantizers.values():
            quantizer.calibrate_mode(True)

        self.model.eval()
        with torch.no_grad():
            for idx, (images, _) in enumerate(dataloader):
                if idx >= num_batches:
                    break
                images = images.to(device)
                _ = self.model(images)

        # Finalize calibration
        for quantizer in self.activation_quantizers.values():
            quantizer.calibrate_mode(False)

    def remove_hooks(self):
        """Remove all activation hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def validate_mixed_precision_config(model_name, checkpoint_path, config_path,
                                      dataset='cifar10', output_json='validation_results.json',
                                      calibration_batches=50):
    """
    Validate a mixed-precision configuration by applying both weight
    and activation quantization and measuring accuracy.

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        config_path: Path to mixed-precision config JSON
        dataset: Dataset name
        output_json: Output results JSON file
        calibration_batches: Number of batches for activation calibration

    Returns:
        Validation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"MIXED-PRECISION VALIDATION (Weight + Activation)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Config: {config_path}")
    print(f"Calibration batches: {calibration_batches}")
    print(f"{'='*60}\n")

    # Initialize timing
    overall_start = time.time()
    timing = {}

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load config
    print("Loading configuration...")
    config_load_start = time.time()
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Handle nested config format
    if 'config' in config_data:
        config = config_data['config']
    else:
        config = config_data

    timing['config_load'] = time.time() - config_load_start
    print(f"  Loaded config for {len(config)} layers")

    # Determine num_classes
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # Load model
    print("\nLoading model...")
    model_load_start = time.time()
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    timing['model_load'] = time.time() - model_load_start
    print(f"  Model loaded in {timing['model_load']:.2f}s")

    # Determine input size and batch size
    if dataset == 'gtsrb':
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:
        input_size = 224

    if model_name == 'swin':
        batch_size = 16
    elif model_name == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load data
    print(f"\nLoading {dataset} data...")
    if dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Measure baseline accuracy (before any quantization)
    print("\n--- Baseline Measurement ---")
    baseline_start = time.time()

    # Reload model for clean baseline
    baseline_model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device=device)
    timing['baseline'] = time.time() - baseline_start
    print(f"Baseline accuracy: {baseline_acc:.2f}% (measured in {timing['baseline']:.2f}s)")
    del baseline_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Create validator
    print("\n--- Applying Mixed-Precision Quantization ---")
    validator = MixedPrecisionValidator(model, config)

    # Apply weight quantization
    print("Applying weight quantization...")
    quant_start = time.time()
    validator.apply_weight_quantization()
    timing['weight_quantization'] = time.time() - quant_start
    print(f"  Weight quantization applied in {timing['weight_quantization']:.2f}s")

    # Setup and calibrate activation quantizers
    print("Setting up activation quantizers...")
    validator.setup_activation_quantizers()
    print(f"  {len(validator.activation_quantizers)} activation quantizers installed")

    print(f"Calibrating ({calibration_batches} batches)...")
    calib_start = time.time()
    validator.calibrate(train_loader, device=device, num_batches=calibration_batches)
    timing['calibration'] = time.time() - calib_start
    print(f"  Calibration completed in {timing['calibration']:.2f}s")

    # Evaluate quantized model
    print("\n--- Evaluating Quantized Model ---")
    eval_start = time.time()
    quantized_acc = evaluate_accuracy(model, test_loader, device=device)
    timing['evaluation'] = time.time() - eval_start
    print(f"Quantized accuracy: {quantized_acc:.2f}% (measured in {timing['evaluation']:.2f}s)")

    # Cleanup
    validator.remove_hooks()

    # Calculate total time
    total_time = time.time() - overall_start
    timing['total'] = total_time

    # Calculate accuracy drop
    acc_drop = baseline_acc - quantized_acc

    # Prepare results
    results = {
        'model': model_name,
        'dataset': dataset,
        'baseline_accuracy': round(baseline_acc, 2),
        'quantized_accuracy': round(quantized_acc, 2),
        'accuracy_drop': round(acc_drop, 2),
        'config_file': config_path,
        'num_layers_configured': len(config),
        'calibration_batches': calibration_batches,
        'timing': {k: round(v, 2) for k, v in timing.items()}
    }

    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_json}")
    print(f"\n  Baseline accuracy:   {baseline_acc:.2f}%")
    print(f"  Quantized accuracy:  {quantized_acc:.2f}%")
    print(f"  Accuracy drop:       {acc_drop:.2f}%")

    # Timing summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Config loading:       {timing.get('config_load', 0):>8.2f}s")
    print(f"  Model loading:        {timing.get('model_load', 0):>8.2f}s")
    print(f"  Weight quantization:  {timing.get('weight_quantization', 0):>8.2f}s")
    print(f"  Calibration:          {timing.get('calibration', 0):>8.2f}s")
    print(f"  Baseline eval:        {timing.get('baseline', 0):>8.2f}s")
    print(f"  Quantized eval:       {timing.get('evaluation', 0):>8.2f}s")
    print(f"  {'-'*40}")
    print(f"  Total time:           {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate Mixed-Precision Config')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to mixed-precision config JSON')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--output', type=str, default='validation_results.json',
                        help='Output results JSON file')
    parser.add_argument('--calibration-batches', type=int, default=50,
                        help='Number of batches for activation calibration')

    args = parser.parse_args()

    validate_mixed_precision_config(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset=args.dataset,
        output_json=args.output,
        calibration_batches=args.calibration_batches
    )
