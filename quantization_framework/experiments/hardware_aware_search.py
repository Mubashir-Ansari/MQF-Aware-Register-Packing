"""
Hardware-Aware Mixed-Precision Search
Assigns bit-widths from user-specified choices based on layer sensitivity.
Architecture-agnostic: Works for VGG, ResNet, LeViT, Swin, and any other models.
"""

import argparse
import json
import csv
import time
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from evaluation.pipeline import get_cifar10_dataloader, get_cifar100_dataloader, evaluate_accuracy


def is_conv_layer(model, layer_name):
    """
    Check if a layer is a convolutional layer (sensitive to extreme quantization).
    Uses module type inspection instead of name matching for architecture-agnostic detection.
    
    Args:
        model: PyTorch model
        layer_name: Layer name to check
    
    Returns:
        True if Conv2d/Conv1d/Conv3d, False otherwise
    """
    for name, module in model.named_modules():
        if name == layer_name:
            # Conv layers are sensitive to 2-bit quantization
            return isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d))
    return False


def load_sensitivity_profile(profile_csv):
    """
    Load sensitivity scores from CSV.
    Returns dict: {layer_name: {bit_width: sensitivity_score}}
    """
    layer_data = {}
    
    with open(profile_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_name = row['layer']
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            # Read sensitivity for different bit-widths
            for key in row.keys():
                if key.startswith('sensitivity_') and 'bit' in key:
                    # Extract bit-width from column name (e.g., 'sensitivity_2bit' -> 2)
                    try:
                        bits = int(key.split('_')[1].replace('bit', ''))
                        sensitivity = float(row[key])
                        layer_data[layer_name][bits] = sensitivity
                    except (ValueError, IndexError):
                        continue
    
    return layer_data


def estimate_accuracy_impact(layer_sensitivities, current_config, layer_name, new_bits, baseline_bits=32):
    """
    Estimate accuracy impact of changing a layer's bit-width.
    
    Simple model: Accuracy drop ≈ sensitivity_score
    
    Args:
        layer_sensitivities: Dict of sensitivity scores per bit-width
        current_config: Current bit-width configuration
        layer_name: Layer to modify
        new_bits: New bit-width to try
        baseline_bits: FP32 baseline (default: 32)
    
    Returns:
        estimated_accuracy_drop: Percentage points
    """
    if layer_name not in layer_sensitivities or new_bits not in layer_sensitivities[layer_name]:
        # No data for this bit-width, use conservative estimate
        # Assume linear relationship: less bits = more drop
        bit_reduction = (baseline_bits - new_bits) / baseline_bits
        return bit_reduction * 5.0  # Conservative 5% max per layer
    
    sensitivity = layer_sensitivities[layer_name][new_bits]
    
    # Sensitivity score represents accuracy drop at this bit-width
    # Higher sensitivity = more accuracy loss
    return sensitivity


def greedy_search(model_name, checkpoint_path, dataset, sensitivity_profile, 
                  bit_choices, target_drop=3.0, baseline_acc=None):
    """
    Greedy search for optimal mixed-precision configuration.
    Architecture-agnostic: Uses module type inspection for layer classification.
    
    Strategy:
    1. Start with all layers at highest bit-width (safest)
    2. Iteratively reduce bit-widths of least sensitive layers
    3. Filter out 2-bit for Conv layers (they can't handle it)
    4. Stop when target accuracy drop is reached
    
    Args:
        model_name: Model architecture name
        checkpoint_path: Path to checkpoint
        dataset: Dataset name
        sensitivity_profile: {layer_name: {bits: sensitivity}}
        bit_choices: User-specified bit-widths (e.g., [2, 4, 8])
        target_drop: Maximum acceptable accuracy drop (default: 3.0%)
        baseline_acc: Baseline FP32 accuracy (if None, will measure)
    
    Returns:
        config: Dict mapping layer names to bit-widths
        final_acc: Estimated final accuracy
    """
    print(f"\n{'='*60}")
    print(f"HARDWARE-AWARE GREEDY SEARCH")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"User bit-widths: {bit_choices}")
    print(f"Target accuracy drop: {target_drop}%")
    print(f"{'='*60}\n")

    # Initialize timing
    search_start = time.time()

    # Sort bit choices (high to low for initialization)
    sorted_bits = sorted(bit_choices, reverse=True)
    highest_bits = sorted_bits[0]
    
    # Initialize: All layers start at highest bit-width
    config = {}
    for layer_name in sensitivity_profile.keys():
        config[layer_name] = highest_bits
    
    print(f"Starting configuration: All layers at {highest_bits}-bit")
    
    # Load model for layer type checking and baseline measurement
    print(f"\nLoading model for analysis...")
    num_classes = 100 if dataset == 'cifar100' else 43 if dataset == 'gtsrb' else 10
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    
    # Measure baseline if not provided
    if baseline_acc is None:
        print(f"Measuring baseline accuracy...")
        baseline_start = time.time()

        # Determine input size based on dataset first, then model
        if dataset == 'gtsrb':
            input_size = 224  # GTSRB uses 224x224 for all models
        elif model_name in ['vgg11_bn', 'resnet']:
            input_size = 32  # CIFAR-10/100 default for these models
        else:
            input_size = 224  # levit, swin, etc.

        # Set batch size based on model type to avoid GPU OOM
        if model_name == 'swin':
            batch_size = 16
        elif model_name == 'levit':
            batch_size = 32
        else:
            batch_size = 128

        if dataset == 'cifar100':
            loader = get_cifar100_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        elif dataset == 'gtsrb':
            from evaluation.pipeline import get_gtsrb_dataloader
            loader = get_gtsrb_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        else:
            loader = get_cifar10_dataloader(train=False, input_size=input_size, batch_size=batch_size)

        baseline_acc = evaluate_accuracy(model, loader, max_samples=1000)
        baseline_time = time.time() - baseline_start
        print(f"Baseline accuracy: {baseline_acc:.2f}% (measured in {baseline_time:.2f}s)")
    
    target_acc = baseline_acc - target_drop
    print(f"Target accuracy: {target_acc:.2f}% (max drop: {target_drop}%)\n")
    
    # Create list of (layer, current_bits, possible_reductions)
    # Sort by sensitivity (least sensitive first for aggressive reduction)
    layer_priorities = []
    
    print("Analyzing layer types...")
    conv_count = 0
    linear_count = 0
    
    for layer_name in config.keys():
        current_bits = config[layer_name]
        
        # Find all lower bit-widths we can try
        possible_reductions = [b for b in sorted_bits if b < current_bits]
        
        # ARCHITECTURE-AGNOSTIC: Filter 2-bit for Conv layers
        if is_conv_layer(model, layer_name):
            if 2 in possible_reductions:
                possible_reductions = [b for b in possible_reductions if b >= 4]
                # print(f"  Conv layer {layer_name}: excluding 2-bit")
            conv_count += 1
        else:
            linear_count += 1
        
        if not possible_reductions:
            continue  # Already at lowest bit-width
        
        # Get average sensitivity across available bit-widths
        sensitivities = []
        for bits in possible_reductions:
            if bits in sensitivity_profile[layer_name]:
                sensitivities.append(sensitivity_profile[layer_name][bits])
        
        if sensitivities:
            avg_sensitivity = sum(sensitivities) / len(sensitivities)
        else:
            avg_sensitivity = 0.5  # Default if no data
        
        layer_priorities.append((layer_name, avg_sensitivity, possible_reductions))
    
    print(f"Detected: {conv_count} Conv layers (min 4-bit), {linear_count} Linear layers (can use 2-bit)\n")
    
    # Clean up model to save memory (we only needed it for layer type checking)
    del model
    torch.cuda.empty_cache()
    
    # Sort by sensitivity (ascending - least sensitive first)
    layer_priorities.sort(key=lambda x: x[1])
    
    print(f"Layer prioritization (least → most sensitive):")
    for i, (layer, sens, _) in enumerate(layer_priorities[:5]):
        print(f"  {i+1}. {layer}: sensitivity={sens:.4f}")
    if len(layer_priorities) > 5:
        print(f"  ... ({len(layer_priorities)} total layers)\n")
    
    # Greedy reduction phase
    cumulative_drop = 0.0
    moves = 0
    greedy_start = time.time()

    print("Starting greedy bit-width reduction...\n")
    
    for layer_name, avg_sensitivity, possible_reductions in layer_priorities:
        current_bits = config[layer_name]
        best_bits = current_bits  # Track the best (lowest) bit-width that fits
        best_drop = 0.0
        reduction_found = False
        
        # Try ALL lower bit-widths (from highest to lowest)
        # Keep trying to find the lowest bit-width that still fits the budget
        for new_bits in sorted(possible_reductions, reverse=True):
            # Estimate accuracy impact
            estimated_drop = estimate_accuracy_impact(
                sensitivity_profile, config, layer_name, new_bits
            )
            
            # Check if we can afford this reduction
            if cumulative_drop + estimated_drop <= target_drop:
                # This bit-width works! Keep trying lower ones
                best_bits = new_bits
                best_drop = estimated_drop
                reduction_found = True
                # DON'T break here - keep trying lower bit-widths!
            else:
                # Can't reduce further without exceeding target
                if moves == 0 and not reduction_found:  # Debug info for first layer only
                    print(f"  [DEBUG] Cannot reduce {layer_name}: "
                          f"estimated_drop={estimated_drop:.2f}% would exceed "
                          f"budget (cumulative={cumulative_drop:.2f}%, target={target_drop:.2f}%)")
        
        # Apply the best (lowest) bit-width we found
        if reduction_found:
            old_bits = config[layer_name]
            config[layer_name] = best_bits
            cumulative_drop += best_drop
            moves += 1
            
            if moves % 5 == 0 or moves <= 10:  # Show first 10 moves + every 5th
                print(f"  Move {moves}: {layer_name} "
                      f"{old_bits}→{best_bits}bit, "
                      f"est_drop={best_drop:.2f}%, "
                      f"cumulative={cumulative_drop:.2f}%")
        
        # Stop if we're at target
        if cumulative_drop >= target_drop * 0.95:  # Within 95% of target
            print(f"\n  Reached target drop ({cumulative_drop:.2f}% ≈ {target_drop}%)")
            break
    
    # Calculate final statistics
    estimated_final_acc = baseline_acc - cumulative_drop
    
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Moves made: {moves}")
    print(f"Estimated accuracy: {estimated_final_acc:.2f}%")
    print(f"Estimated drop: {cumulative_drop:.2f}%")
    
    # Bit-width distribution
    bit_distribution = {}
    for bits in bit_choices:
        count = sum(1 for b in config.values() if b == bits)
        bit_distribution[bits] = count
    
    print(f"\nBit-width distribution:")
    for bits in sorted(bit_distribution.keys(), reverse=True):
        count = bit_distribution[bits]
        percentage = (count / len(config)) * 100
        print(f"  {bits}-bit: {count:3d} layers ({percentage:5.1f}%)")
    
    # Calculate compression ratio
    total_bits_original = len(config) * 32  # FP32 baseline
    total_bits_quantized = sum(config.values())
    compression_ratio = total_bits_original / total_bits_quantized
    
    print(f"\nEstimated compression: {compression_ratio:.2f}x")

    # Calculate timing
    greedy_time = time.time() - greedy_start
    total_time = time.time() - search_start

    print(f"\n{'-'*40}")
    print(f"TIMING:")
    print(f"  Greedy search:  {greedy_time:>8.2f}s")
    print(f"  Total time:     {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    return config, estimated_final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hardware-Aware Mixed-Precision Search')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--profile', type=str, required=True, help='Sensitivity profile CSV')
    parser.add_argument('--output', type=str, required=True, help='Output config JSON')
    parser.add_argument('--bits', type=int, nargs='+', required=True, 
                        help='User-specified bit-widths (e.g., --bits 2 4 8)')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='Dataset (cifar10/cifar100/gtsrb)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Target accuracy drop percentage (default: 3.0)')
    parser.add_argument('--baseline-acc', type=float, default=None,
                        help='Baseline accuracy (if known, to skip measurement)')
    
    args = parser.parse_args()
    
    # Load sensitivity profile
    print(f"Loading sensitivity profile: {args.profile}")
    sensitivity_profile = load_sensitivity_profile(args.profile)
    print(f"Loaded {len(sensitivity_profile)} layers\n")
    
    # Run greedy search
    config, final_acc = greedy_search(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        sensitivity_profile=sensitivity_profile,
        bit_choices=args.bits,
        target_drop=args.target_drop,
        baseline_acc=args.baseline_acc
    )
    
    # Save configuration
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {args.output}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Validate PTQ accuracy:")
    print(f"   python quantization_framework/experiments/validate_config.py \\")
    print(f"     --model {args.model} \\")
    print(f"     --checkpoint {args.checkpoint} \\")
    print(f"     --config {args.output} \\")
    print(f"     --dataset {args.dataset}")
    print(f"\n2. If PTQ fails, run QAT:")
    print(f"   python quantization_framework/experiments/qat_training.py \\")
    print(f"     --model {args.model} \\")
    print(f"     --checkpoint {args.checkpoint} \\")
    print(f"     --config {args.output} \\")
    print(f"     --dataset {args.dataset}")
    print(f"{'='*60}\n")
