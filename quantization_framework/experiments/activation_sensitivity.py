"""
Per-Layer Activation Sensitivity Analysis
==========================================

Tests activation quantization for each layer individually.

Methodology:
  - For EACH layer:
      1. Register forward hook that quantizes that layer's output activations
      2. Keep all other layers at FP32
      3. Measure accuracy drop
      4. Remove hook

Output: CSV with per-layer activation sensitivity scores

Usage:
    python activation_sensitivity.py \
        --model vgg11_bn \
        --checkpoint models/vgg11_bn.pt \
        --dataset cifar10 \
        --output vgg_activation_sensitivity.csv \
        --bits 2 4 6 8
"""

import argparse
import csv
import time
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor


def make_quantize_hook(bit_width):
    """
    Create forward hook that quantizes output activations.

    Args:
        bit_width: Number of bits for quantization

    Returns:
        Hook function
    """
    def hook(module, input, output):
        # Quantize output activations
        q_output, _, _ = quantize_tensor(output, bit_width=bit_width, method='symmetric')
        return q_output
    return hook


def measure_activation_sensitivity(model, dataloader, bit_widths=[2, 4, 6, 8],
                                   device='cuda', max_samples=2000):
    """
    Measure per-layer activation sensitivity.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        bit_widths: List of bit-widths to test
        device: Device to run on
        max_samples: Number of samples for evaluation

    Returns:
        results: List of dicts with per-layer results
        baseline_acc: Baseline FP32 accuracy
    """
    print("\n" + "="*70)
    print("ACTIVATION SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Testing bit-widths: {bit_widths}")
    print(f"Max samples per test: {max_samples}")
    print("="*70)

    model.to(device)
    model.eval()

    # Measure baseline accuracy (FP32)
    print("\n[Step 1/2] Measuring baseline accuracy (FP32)...")
    baseline_start = time.time()
    baseline_acc = evaluate_accuracy(model, dataloader, device=device, max_samples=max_samples)
    baseline_time = time.time() - baseline_start

    print(f"Baseline: {baseline_acc:.2f}% (took {baseline_time:.1f}s)")

    # Get all layers to test
    layers_to_test = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers_to_test.append((name, module))

    print(f"\n[Step 2/2] Testing {len(layers_to_test)} layers...")
    print("="*70)

    results = []
    total_tests = len(layers_to_test) * len(bit_widths)
    current_test = 0

    for layer_idx, (layer_name, layer_module) in enumerate(layers_to_test):
        print(f"\n[{layer_idx+1}/{len(layers_to_test)}] Layer: {layer_name}")
        print(f"  Type: {type(layer_module).__name__}")

        layer_results = {
            'layer': layer_name,
            'layer_type': type(layer_module).__name__,
            'baseline_acc': baseline_acc
        }

        # Test each bit-width
        for bits in sorted(bit_widths):
            current_test += 1
            progress = (current_test / total_tests) * 100

            print(f"  Testing {bits}-bit... ", end="", flush=True)

            test_start = time.time()

            # Register hook on this layer
            hook = layer_module.register_forward_hook(make_quantize_hook(bits))

            try:
                # Evaluate with quantized activations
                acc = evaluate_accuracy(model, dataloader, device=device, max_samples=max_samples)
                drop = baseline_acc - acc

                # Store results
                layer_results[f'accuracy_{bits}bit'] = round(acc, 2)
                layer_results[f'sensitivity_{bits}bit'] = round(drop, 2)

                test_time = time.time() - test_start

                # Status indicator
                if drop < 1.0:
                    status = "OK"
                elif drop < 5.0:
                    status = "MODERATE"
                elif drop < 20.0:
                    status = "HIGH"
                else:
                    status = "CRITICAL"

                print(f"Acc: {acc:.2f}% (drop: {drop:.2f}%) [{status}] ({test_time:.1f}s)")

            except Exception as e:
                print(f"Error: {e}")
                layer_results[f'accuracy_{bits}bit'] = 0.0
                layer_results[f'sensitivity_{bits}bit'] = 100.0

            finally:
                # Always remove hook
                hook.remove()

        results.append(layer_results)

    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)

    return results, baseline_acc


def save_results(results, output_file, bit_widths):
    """Save results to CSV."""
    if not results:
        print("No results to save!")
        return

    # Build fieldnames
    fieldnames = ['layer', 'layer_type', 'baseline_acc']
    for bits in sorted(bit_widths):
        fieldnames.extend([f'accuracy_{bits}bit', f'sensitivity_{bits}bit'])

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


def print_summary(results, bit_widths):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if not results:
        print("No results!")
        return

    # Find most sensitive layers at lowest bit-width
    min_bits = min(bit_widths)

    # Sort by sensitivity at min_bits
    sorted_results = sorted(
        results,
        key=lambda x: x.get(f'sensitivity_{min_bits}bit', 0),
        reverse=True
    )

    print(f"\nTop 10 Most Sensitive Layers (at {min_bits}-bit activations):")
    for i, result in enumerate(sorted_results[:10], 1):
        layer = result['layer']
        sensitivity = result.get(f'sensitivity_{min_bits}bit', 0)
        print(f"  {i:2d}. {layer:30s}: {sensitivity:6.2f}% drop")

    # Find most robust layers
    print(f"\nTop 10 Most Robust Layers (at {min_bits}-bit activations):")
    robust_sorted = sorted(sorted_results, key=lambda x: x.get(f'sensitivity_{min_bits}bit', 100))
    for i, result in enumerate(robust_sorted[:10], 1):
        layer = result['layer']
        sensitivity = result.get(f'sensitivity_{min_bits}bit', 0)
        print(f"  {i:2d}. {layer:30s}: {sensitivity:6.2f}% drop")

    # Statistics
    sensitivities = [r.get(f'sensitivity_{min_bits}bit', 0) for r in results]
    avg_sensitivity = sum(sensitivities) / len(sensitivities)
    max_sensitivity = max(sensitivities)
    min_sensitivity = min(sensitivities)

    print(f"\nStatistics (at {min_bits}-bit):")
    print(f"  Average sensitivity: {avg_sensitivity:.2f}%")
    print(f"  Max sensitivity: {max_sensitivity:.2f}%")
    print(f"  Min sensitivity: {min_sensitivity:.2f}%")

    # Count by category
    critical = sum(1 for s in sensitivities if s > 20.0)
    high = sum(1 for s in sensitivities if 5.0 < s <= 20.0)
    moderate = sum(1 for s in sensitivities if 1.0 < s <= 5.0)
    robust = sum(1 for s in sensitivities if s <= 1.0)

    print(f"\nLayer Categories:")
    print(f"  Critical (>20% drop): {critical} layers")
    print(f"  High (5-20% drop): {high} layers")
    print(f"  Moderate (1-5% drop): {moderate} layers")
    print(f"  Robust (<1% drop): {robust} layers")

    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-Layer Activation Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python activation_sensitivity.py \\
      --model vgg11_bn \\
      --checkpoint models/vgg11_bn.pt \\
      --dataset cifar10 \\
      --output vgg_activation_sensitivity.csv

  # Test specific bit-widths
  python activation_sensitivity.py \\
      --model swin \\
      --checkpoint models/best_swin_model.pth \\
      --dataset cifar100 \\
      --bits 4 6 8 \\
      --output swin_activation_sensitivity.csv
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (vgg11_bn, swin, resnet, levit)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'gtsrb'],
                       help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: {model}_activation_sensitivity.csv)')

    # Optional arguments
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 6, 8],
                       help='Bit-widths to test (default: 2 4 6 8)')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Max samples per evaluation (default: 2000)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"{args.model}_activation_sensitivity.csv"
        print(f"No output file specified, using: {args.output}")

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)

    # Setup
    print("="*70)
    print("SETUP")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Bit-widths: {args.bits}")
    print("="*70)

    # Determine number of classes and input size
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # Input size based on model
    if args.model in ['swin', 'levit']:
        input_size = 224
    else:
        input_size = 32

    # Batch size based on model
    if args.model == 'swin':
        batch_size = 16
    elif args.model == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load dataloader
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'cifar100':
        dataloader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        dataloader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        dataloader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Load model
    print(f"Loading model...")
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)

    if model is None:
        print(f"Error: Failed to load model")
        exit(1)

    print(f"Model loaded successfully")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Run sensitivity analysis
    start_time = time.time()

    results, baseline_acc = measure_activation_sensitivity(
        model=model,
        dataloader=dataloader,
        bit_widths=args.bits,
        device=args.device,
        max_samples=args.max_samples
    )

    total_time = time.time() - start_time

    # Save results
    save_results(results, args.output, args.bits)

    # Print summary
    print_summary(results, args.bits)

    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Output: {args.output}")
