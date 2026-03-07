"""
Mixed-Precision Search for Joint Weight + Activation Quantization

Uses ACTUAL MODEL EVALUATION to find optimal per-layer bit-widths
for BOTH weights AND activations via greedy search.

Usage:
    python mixed_precision_search.py \
        --model vgg11_bn \
        --checkpoint path/to/checkpoint.pth \
        --dataset cifar10 \
        --target-acc-drop 1.0 \
        --output mixed_config.json
"""

import argparse
import json
import csv
import time
import sys
import os
import copy

import torch
import torch.nn as nn

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


def load_weight_sensitivity(csv_path):
    """
    Load weight sensitivity profile from CSV.

    Expected CSV format:
        layer, baseline_acc, accuracy_2bit, sensitivity_2bit, accuracy_4bit, ...

    Returns:
        dict: {layer_name: {2: sensitivity, 4: sensitivity, 8: sensitivity}}
    """
    sensitivity = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_name = row['layer']
            sensitivity[layer_name] = {}

            for key, value in row.items():
                if key.startswith('sensitivity_') and key.endswith('bit'):
                    bit_width = int(key.replace('sensitivity_', '').replace('bit', ''))
                    sensitivity[layer_name][bit_width] = float(value)

    return sensitivity


def load_activation_sensitivity(json_path):
    """
    Load activation sensitivity profile from JSON.

    Returns:
        dict: {layer_name: {4: drop, 6: drop, 8: drop}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    sensitivity = {}
    for layer_name, layer_data in data.get('layers', {}).items():
        sensitivity[layer_name] = {}
        for key, value in layer_data.items():
            if key.startswith('drop_') and key.endswith('bit'):
                bit_width = int(key.replace('drop_', '').replace('bit', ''))
                sensitivity[layer_name][bit_width] = value

    return sensitivity


class CalibrationActivationQuantizer(nn.Module):
    """Activation quantizer with calibration support for search."""
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.calibrating = False
        self.calibrated = False
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def calibrate_mode(self, enable=True):
        self.calibrating = enable
        if not enable and self.min_val.item() < float('inf'):
            self._compute_params()
            self.calibrated = True

    def _compute_params(self):
        q_min, q_max = 0, 2 ** self.bit_width - 1
        min_val, max_val = self.min_val.item(), self.max_val.item()
        scale = max((max_val - min_val) / (q_max - q_min), 1e-8)
        zero_point = max(q_min, min(q_max, round(-min_val / scale)))
        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)

    def forward(self, x):
        if self.scale.device != x.device:
            self.min_val = self.min_val.to(x.device)
            self.max_val = self.max_val.to(x.device)
            self.scale = self.scale.to(x.device)
            self.zero_point = self.zero_point.to(x.device)

        if self.calibrating:
            with torch.no_grad():
                self.min_val = torch.min(self.min_val, x.min())
                self.max_val = torch.max(self.max_val, x.max())
            return x

        if not self.calibrated:
            return x

        q_min, q_max = 0, 2 ** self.bit_width - 1
        x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
        return (x_int - self.zero_point) * self.scale


class MixedPrecisionEvaluator:
    """Evaluates a model with mixed-precision W+A quantization."""

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.activation_quantizers = {}
        self.hooks = []

    def apply_weight_quantization(self):
        """Apply weight quantization according to config."""
        for name, module in self.model.named_modules():
            if name in self.config and hasattr(module, 'weight') and module.weight is not None:
                cfg = self.config[name]
                bits = cfg['weight'] if isinstance(cfg, dict) else cfg
                w = module.weight.data
                q_w, _, _ = quantize_tensor(w, bit_width=bits)
                module.weight.data = q_w

    def setup_activation_quantizers(self):
        """Install activation quantizers as forward hooks."""
        for name, module in self.model.named_modules():
            if name in self.config:
                cfg = self.config[name]
                if isinstance(cfg, dict) and 'activation' in cfg:
                    bits = cfg['activation']
                    quantizer = CalibrationActivationQuantizer(bit_width=bits)
                    self.activation_quantizers[name] = quantizer

                    def make_hook(quant):
                        def hook(mod, inp, out):
                            return quant(out)
                        return hook

                    handle = module.register_forward_hook(make_hook(quantizer))
                    self.hooks.append(handle)

    def calibrate(self, dataloader, num_batches=20):
        """Calibrate activation quantizers."""
        for q in self.activation_quantizers.values():
            q.calibrate_mode(True)

        self.model.eval()
        with torch.no_grad():
            for idx, (images, _) in enumerate(dataloader):
                if idx >= num_batches:
                    break
                images = images.to(self.device)
                _ = self.model(images)

        for q in self.activation_quantizers.values():
            q.calibrate_mode(False)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activation_quantizers = {}


def calculate_bops(model, config, input_size=32):
    """
    Calculate Bit Operations (BOPs) for a mixed-precision configuration.

    BOPs = sum over layers of: MACs * weight_bits * activation_bits

    Returns:
        BOPs in GigaBOPs (GBOPs)
    """
    total_bops = 0

    for name, module in model.named_modules():
        if name in config:
            cfg = config[name]
            w_bits = cfg['weight'] if isinstance(cfg, dict) else cfg
            a_bits = cfg.get('activation', 8) if isinstance(cfg, dict) else 8

            if isinstance(module, nn.Conv2d):
                # MACs = H_out * W_out * C_in * C_out * K^2
                # Simplified: assume output size ≈ input size (stride=1, same padding)
                h_out = input_size // (module.stride[0] if hasattr(module.stride, '__getitem__') else module.stride)
                w_out = h_out
                macs = h_out * w_out * module.in_channels * module.out_channels * \
                       module.kernel_size[0] * module.kernel_size[1]
                total_bops += macs * w_bits * a_bits

            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
                total_bops += macs * w_bits * a_bits

    return total_bops / 1e9  # Convert to GBOPs


def evaluate_config(model_name, checkpoint_path, config, train_loader, test_loader,
                    device='cuda', num_classes=10, calibration_batches=20, max_samples=2000):
    """
    Evaluate a mixed-precision configuration with ACTUAL model evaluation.

    Returns:
        accuracy (float): Accuracy percentage
    """
    # Load fresh model
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    # Create evaluator and apply quantization
    evaluator = MixedPrecisionEvaluator(model, config, device)
    evaluator.apply_weight_quantization()
    evaluator.setup_activation_quantizers()
    evaluator.calibrate(train_loader, num_batches=calibration_batches)

    # Evaluate
    acc = evaluate_accuracy(model, test_loader, device=device, max_samples=max_samples)

    # Cleanup
    evaluator.remove_hooks()
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return acc


def search_mixed_precision_config(model_name, checkpoint_path, dataset='cifar10',
                                   target_acc_drop=1.0, output_json='mixed_config.json',
                                   weight_bits_options=[2, 4, 8],
                                   activation_bits_options=[4, 6, 8],
                                   use_sensitivity_hints=True,
                                   weight_sensitivity_path=None,
                                   activation_sensitivity_path=None,
                                   calibration_batches=20,
                                   eval_samples=2000):
    """
    Greedy search for optimal joint W+A bit-width configuration using ACTUAL evaluation.

    Algorithm:
    1. Start with max precision (8-bit W, 8-bit A)
    2. Iteratively try reducing bit-widths for each layer
    3. Accept reduction if accuracy stays within budget
    4. Use sensitivity hints (if available) to prioritize layers

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name
        target_acc_drop: Maximum acceptable accuracy drop (%)
        output_json: Output JSON file path
        weight_bits_options: Weight bit-width options
        activation_bits_options: Activation bit-width options
        use_sensitivity_hints: Use sensitivity profiles to guide search order
        weight_sensitivity_path: Optional path to weight sensitivity CSV
        activation_sensitivity_path: Optional path to activation sensitivity JSON
        calibration_batches: Batches for activation calibration
        eval_samples: Max samples for evaluation (for speed)

    Returns:
        Configuration dict: {layer: {"weight": bits, "activation": bits}}
    """
    print(f"\n{'='*60}")
    print(f"MIXED-PRECISION SEARCH (Weight + Activation)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Target accuracy drop: {target_acc_drop}%")
    print(f"Weight bit options: {weight_bits_options}")
    print(f"Activation bit options: {activation_bits_options}")
    print(f"{'='*60}\n")

    # Initialize timing
    overall_start = time.time()
    timing = {'baseline': 0, 'search_iterations': [], 'total': 0}

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Determine num_classes and input_size
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    if dataset == 'gtsrb':
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:
        input_size = 224

    # Batch size
    if model_name == 'swin':
        batch_size = 16
    elif model_name == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load data
    print("Loading data...")
    if dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Get quantizable layers
    print("Identifying quantizable layers...")
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append(name)
    del model
    print(f"Found {len(layers)} quantizable layers\n")

    # Initialize config with max precision
    max_w_bits = max(weight_bits_options)
    max_a_bits = max(activation_bits_options)
    config = {layer: {'weight': max_w_bits, 'activation': max_a_bits} for layer in layers}

    # Measure baseline accuracy (with max precision quantization)
    print("Measuring baseline (max precision)...")
    baseline_start = time.time()
    baseline_acc = evaluate_config(
        model_name, checkpoint_path, config, train_loader, test_loader,
        device, num_classes, calibration_batches, eval_samples
    )
    timing['baseline'] = time.time() - baseline_start
    print(f"Baseline accuracy (W{max_w_bits}/A{max_a_bits}): {baseline_acc:.2f}% [{timing['baseline']:.1f}s]\n")

    min_acceptable_acc = baseline_acc - target_acc_drop

    # Load sensitivity hints if available
    layer_priorities = {layer: 0 for layer in layers}
    if use_sensitivity_hints:
        if weight_sensitivity_path and os.path.exists(weight_sensitivity_path):
            w_sens = load_weight_sensitivity(weight_sensitivity_path)
            for layer in layers:
                if layer in w_sens:
                    # Higher sensitivity = lower priority (process last)
                    layer_priorities[layer] += max(w_sens[layer].values(), default=0)

        if activation_sensitivity_path and os.path.exists(activation_sensitivity_path):
            a_sens = load_activation_sensitivity(activation_sensitivity_path)
            for layer in layers:
                if layer in a_sens:
                    layer_priorities[layer] += max(a_sens[layer].values(), default=0)

    # Sort layers: least sensitive first (can compress more aggressively)
    sorted_layers = sorted(layers, key=lambda x: layer_priorities[x])

    # Greedy search
    print("Running greedy search with ACTUAL evaluation...\n")
    search_start = time.time()
    iteration = 0
    improvements_made = True

    while improvements_made:
        improvements_made = False
        iteration += 1
        iter_start = time.time()

        print(f"--- Iteration {iteration} ---")

        for layer in sorted_layers:
            current_w = config[layer]['weight']
            current_a = config[layer]['activation']

            # Try reducing weight bits
            lower_w_options = [b for b in weight_bits_options if b < current_w]
            for new_w in sorted(lower_w_options, reverse=True):  # Try highest reduction first
                test_config = copy.deepcopy(config)
                test_config[layer]['weight'] = new_w

                acc = evaluate_config(
                    model_name, checkpoint_path, test_config, train_loader, test_loader,
                    device, num_classes, calibration_batches, eval_samples
                )

                if acc >= min_acceptable_acc:
                    config[layer]['weight'] = new_w
                    improvements_made = True
                    print(f"  {layer}: W {current_w}->{new_w} bit (acc: {acc:.2f}%)")
                    break

            # Try reducing activation bits
            current_a = config[layer]['activation']
            lower_a_options = [b for b in activation_bits_options if b < current_a]
            for new_a in sorted(lower_a_options, reverse=True):
                test_config = copy.deepcopy(config)
                test_config[layer]['activation'] = new_a

                acc = evaluate_config(
                    model_name, checkpoint_path, test_config, train_loader, test_loader,
                    device, num_classes, calibration_batches, eval_samples
                )

                if acc >= min_acceptable_acc:
                    config[layer]['activation'] = new_a
                    improvements_made = True
                    print(f"  {layer}: A {current_a}->{new_a} bit (acc: {acc:.2f}%)")
                    break

        iter_time = time.time() - iter_start
        timing['search_iterations'].append(iter_time)
        print(f"  Iteration time: {iter_time:.1f}s\n")

        if iteration >= 10:  # Safety limit
            print("Max iterations reached.")
            break

    search_time = time.time() - search_start

    # Final evaluation
    print("Final evaluation...")
    final_acc = evaluate_config(
        model_name, checkpoint_path, config, train_loader, test_loader,
        device, num_classes, calibration_batches, eval_samples
    )
    acc_drop = baseline_acc - final_acc

    # Calculate BOPs
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    bops = calculate_bops(model, config, input_size)
    baseline_bops = calculate_bops(model, {layer: {'weight': 32, 'activation': 32} for layer in layers}, input_size)
    bops_reduction = baseline_bops / bops if bops > 0 else 1.0
    del model

    # Calculate statistics
    total_time = time.time() - overall_start
    timing['search'] = search_time
    timing['total'] = total_time

    w_bits_dist = {}
    a_bits_dist = {}
    for layer, cfg in config.items():
        w_bits_dist[cfg['weight']] = w_bits_dist.get(cfg['weight'], 0) + 1
        a_bits_dist[cfg['activation']] = a_bits_dist.get(cfg['activation'], 0) + 1

    avg_w_bits = sum(cfg['weight'] for cfg in config.values()) / len(config)
    avg_a_bits = sum(cfg['activation'] for cfg in config.values()) / len(config)

    # Prepare output
    output = {
        'model': model_name,
        'dataset': dataset,
        'target_acc_drop': target_acc_drop,
        'baseline_accuracy': round(baseline_acc, 2),
        'final_accuracy': round(final_acc, 2),
        'actual_acc_drop': round(acc_drop, 2),
        'layers': len(config),
        'avg_weight_bits': round(avg_w_bits, 2),
        'avg_activation_bits': round(avg_a_bits, 2),
        'weight_bit_distribution': w_bits_dist,
        'activation_bit_distribution': a_bits_dist,
        'bops_gbops': round(bops, 2),
        'baseline_bops_gbops': round(baseline_bops, 2),
        'bops_reduction': round(bops_reduction, 2),
        'config': config,
        'timing': {k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()}
    }

    # Save configuration
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Configuration saved to: {output_json}")
    print(f"\n  Baseline accuracy:  {baseline_acc:.2f}%")
    print(f"  Final accuracy:     {final_acc:.2f}%")
    print(f"  Accuracy drop:      {acc_drop:.2f}%")
    print(f"\n  Avg weight bits:    {avg_w_bits:.1f}")
    print(f"  Avg activation bits:{avg_a_bits:.1f}")
    print(f"\n  BOPs:               {bops:.2f} GBOPs")
    print(f"  BOPs reduction:     {bops_reduction:.1f}x")
    print(f"\nWeight bit-width distribution:")
    for bits, count in sorted(w_bits_dist.items()):
        print(f"  {bits}-bit: {count} layers")
    print(f"\nActivation bit-width distribution:")
    for bits, count in sorted(a_bits_dist.items()):
        print(f"  {bits}-bit: {count} layers")

    # Timing summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline eval:      {timing['baseline']:>8.2f}s")
    print(f"  Search iterations:  {len(timing['search_iterations']):>8d}")
    print(f"  Total search time:  {search_time:>8.2f}s")
    print(f"  {'-'*40}")
    print(f"  Total time:         {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixed-Precision Search (W+A) with Actual Evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--target-acc-drop', type=float, default=1.0,
                        help='Target accuracy drop budget (%)')
    parser.add_argument('--output', type=str, default='mixed_config.json',
                        help='Output JSON configuration file')
    parser.add_argument('--weight-bits', type=int, nargs='+', default=[2, 4, 8],
                        help='Weight bit-width options')
    parser.add_argument('--activation-bits', type=int, nargs='+', default=[4, 6, 8],
                        help='Activation bit-width options')
    parser.add_argument('--weight-sensitivity', type=str, default=None,
                        help='Optional: weight sensitivity CSV for search hints')
    parser.add_argument('--activation-sensitivity', type=str, default=None,
                        help='Optional: activation sensitivity JSON for search hints')
    parser.add_argument('--calibration-batches', type=int, default=20,
                        help='Batches for activation calibration')
    parser.add_argument('--eval-samples', type=int, default=2000,
                        help='Max samples for each evaluation')

    args = parser.parse_args()

    search_mixed_precision_config(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        target_acc_drop=args.target_acc_drop,
        output_json=args.output,
        weight_bits_options=args.weight_bits,
        activation_bits_options=args.activation_bits,
        use_sensitivity_hints=(args.weight_sensitivity is not None or args.activation_sensitivity is not None),
        weight_sensitivity_path=args.weight_sensitivity,
        activation_sensitivity_path=args.activation_sensitivity,
        calibration_batches=args.calibration_batches,
        eval_samples=args.eval_samples
    )
