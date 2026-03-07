"""
Activation Sensitivity Analysis

Measures per-layer activation sensitivity at different bit-widths (4, 6, 8-bit).
This enables research into mixed-precision activation quantization.

Usage:
    python activation_sensitivity.py --model vgg11_bn --dataset cifar10 --output act_sens.json
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)


class ActivationQuantizer(nn.Module):
    """
    Fake quantization module for activations.
    Quantizes activations to specified bit-width during forward pass.
    """
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x

        # Asymmetric quantization for activations (typically ReLU outputs >= 0)
        q_min = 0
        q_max = 2 ** self.bit_width - 1

        # Compute scale and zero point from tensor statistics
        min_val = x.min()
        max_val = x.max()

        scale = (max_val - min_val) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero

        zero_point = torch.round(-min_val / scale).clamp(q_min, q_max)

        # Fake quantize: quantize then dequantize
        x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
        x_dq = (x_int - zero_point) * scale

        return x_dq


class ActivationSensitivityHook:
    """
    Forward hook that applies fake quantization to layer activations.
    Used to measure accuracy impact of quantizing specific layer's activations.
    """
    def __init__(self, bit_width=8):
        self.quantizer = ActivationQuantizer(bit_width)
        self.handle = None

    def hook_fn(self, module, input, output):
        """Replace output with quantized version."""
        return self.quantizer(output)

    def attach(self, module):
        """Attach hook to a module."""
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_quantizable_activation_layers(model):
    """
    Get layers whose activations can be quantized.
    Returns layers that produce activations (Conv2d, Linear, BatchNorm, ReLU).
    """
    layers = []
    for name, module in model.named_modules():
        # Target layers that produce quantizable activations
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.GELU)):
            # Skip the final classifier layer (keep full precision)
            if 'classifier' in name and isinstance(module, nn.Linear):
                # Check if this is the last linear layer
                continue
            layers.append((name, module))
    return layers


def measure_activation_sensitivity(model_name, checkpoint_path, dataset='cifar10',
                                    output_json='act_sensitivity.json',
                                    bit_widths=[4, 6, 8],
                                    max_samples=2000):
    """
    Measure per-layer activation sensitivity at different bit-widths.

    For each layer, quantize ONLY that layer's activations and measure accuracy.
    This reveals which layers are sensitive to activation quantization.

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name ('cifar10', 'cifar100', 'gtsrb')
        output_json: Output JSON file path
        bit_widths: List of bit-widths to test (default: [4, 6, 8])
        max_samples: Maximum samples for evaluation (for speed)

    Returns:
        Dictionary with sensitivity results
    """
    print(f"\n{'='*60}")
    print(f"ACTIVATION SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Testing bit-widths: {bit_widths}")
    print(f"Max samples: {max_samples}")
    print(f"{'='*60}\n")

    # Initialize timing
    overall_start = time.time()
    layer_times = []

    # Determine num_classes
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # Load model
    print("Loading model...")
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Determine input size based on model and dataset
    if dataset == 'gtsrb':
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:
        input_size = 224  # Swin/LeViT

    # Adjust batch size for memory
    if model_name == 'swin':
        batch_size = 16
    elif model_name == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load data
    print(f"Loading {dataset} data...")
    if dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Clear GPU cache
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Measure baseline accuracy (no activation quantization)
    print("\nMeasuring baseline accuracy...")
    baseline_start = time.time()
    baseline_acc = evaluate_accuracy(model, loader, device=device, max_samples=max_samples)
    baseline_time = time.time() - baseline_start
    print(f"Baseline Accuracy: {baseline_acc:.2f}% (measured in {baseline_time:.2f}s)\n")

    # Get quantizable layers
    layers = get_quantizable_activation_layers(model)
    print(f"Found {len(layers)} layers for activation analysis\n")

    # Store results
    results = {
        'model': model_name,
        'dataset': dataset,
        'baseline_accuracy': baseline_acc,
        'bit_widths_tested': bit_widths,
        'layers': {}
    }

    # Analyze each layer
    for idx, (layer_name, layer_module) in enumerate(layers):
        layer_start = time.time()
        print(f"[{idx+1}/{len(layers)}] Analyzing {layer_name}...")

        layer_result = {
            'type': type(layer_module).__name__,
            'baseline_acc': baseline_acc
        }

        for bit_width in sorted(bit_widths):
            # Create hook for this bit-width
            hook = ActivationSensitivityHook(bit_width=bit_width)
            hook.attach(layer_module)

            # Evaluate with quantized activations
            try:
                acc = evaluate_accuracy(model, loader, device=device, max_samples=max_samples)
            except Exception as e:
                print(f"    Error at {bit_width}-bit: {e}")
                acc = 0.0
            finally:
                hook.remove()

            sensitivity = baseline_acc - acc

            layer_result[f'acc_{bit_width}bit'] = round(acc, 2)
            layer_result[f'drop_{bit_width}bit'] = round(sensitivity, 2)

            print(f"    {bit_width}-bit: Acc={acc:.2f}%, Drop={sensitivity:.2f}%")

        layer_time = time.time() - layer_start
        layer_times.append(layer_time)
        layer_result['analysis_time_seconds'] = round(layer_time, 2)

        results['layers'][layer_name] = layer_result
        print(f"    [TIMING] Layer analyzed in {layer_time:.2f}s\n")

        # Clear GPU memory periodically
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Calculate timing summary
    total_time = time.time() - overall_start
    avg_layer_time = sum(layer_times) / len(layer_times) if layer_times else 0

    # Add timing to results
    results['timing'] = {
        'baseline_seconds': round(baseline_time, 2),
        'layers_analyzed': len(layers),
        'avg_layer_seconds': round(avg_layer_time, 2),
        'total_layer_analysis_seconds': round(sum(layer_times), 2),
        'total_seconds': round(total_time, 2)
    }

    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"{'='*60}")
    print(f"RESULTS SAVED TO: {output_json}")
    print(f"{'='*60}")

    # Find most/least sensitive layers
    sensitivities = []
    for layer_name, layer_data in results['layers'].items():
        if f'drop_4bit' in layer_data:
            sensitivities.append((layer_name, layer_data['drop_4bit']))

    sensitivities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nMost Sensitive Layers (4-bit activation):")
    for name, drop in sensitivities[:5]:
        print(f"  {name}: {drop:.2f}% drop")

    print(f"\nLeast Sensitive Layers (4-bit activation):")
    for name, drop in sensitivities[-5:]:
        print(f"  {name}: {drop:.2f}% drop")

    # Timing summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline measurement:   {baseline_time:>8.2f}s")
    print(f"  Layers analyzed:        {len(layers):>8d}")
    print(f"  Avg time per layer:     {avg_layer_time:>8.2f}s")
    print(f"  Total layer analysis:   {sum(layer_times):>8.2f}s")
    print(f"  {'-'*40}")
    print(f"  Total time:             {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Activation Sensitivity Analysis')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--output', type=str, default='act_sensitivity.json',
                        help='Output JSON file path')
    parser.add_argument('--bits', type=int, nargs='+', default=[4, 6, 8],
                        help='Bit-widths to test (e.g., --bits 4 6 8)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Maximum samples for evaluation')

    args = parser.parse_args()

    measure_activation_sensitivity(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_json=args.output,
        bit_widths=args.bits,
        max_samples=args.max_samples
    )
