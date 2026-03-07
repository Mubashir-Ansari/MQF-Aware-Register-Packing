"""
Generate Mixed-Precision Activation Configuration

Matches activation bit-widths to weight bit-widths based on layer sensitivity.
Architecture-agnostic: Works for VGG, ResNet, LeViT, Swin, and any model.

Usage:
    python generate_activation_config.py \
        --weight-config model_weight_config.json \
        --activation-sensitivity model_activation_sensitivity.csv \
        --output model_activation_config.json \
        --mode safe
"""

import argparse
import json
import csv


def load_activation_sensitivity(csv_path):
    """
    Load activation sensitivity from CSV.
    
    Returns:
        dict: {layer_name: {2bit: drop, 4bit: drop, 6bit: drop, 8bit: drop}}
    """
    sensitivity = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_name = row['layer']
            sensitivity[layer_name] = {}
            
            # Extract bit-width sensitivities
            for key in row.keys():
                if key.startswith('sensitivity_') and 'bit' in key:
                    try:
                        bits = int(key.split('_')[1].replace('bit', ''))
                        drop = float(row[key])
                        sensitivity[layer_name][bits] = drop
                    except (ValueError, IndexError):
                        continue
    
    return sensitivity


def get_activation_bits(layer_name, weight_bits, activation_sensitivity, mode='safe'):
    """
    Determine activation bit-width for a layer.
    
    ARCHITECTURE-AGNOSTIC: Uses sensitivity thresholds, not hardcoded names.
    
    Args:
        layer_name: Layer to quantize
        weight_bits: Weight bit-width assigned to this layer
        activation_sensitivity: Dict from CSV {layer: {2bit: drop, 4bit: drop, ...}}
        mode: 'safe', 'matched', or 'aggressive'
    
    Returns:
        int: Activation bit-width (2, 4, 6, or 8)
    """
    
    # Default to safe if layer not in sensitivity data
    if layer_name not in activation_sensitivity:
        print(f"  [WARNING] {layer_name} not in sensitivity data, defaulting to A8")
        return 8
    
    layer_sens = activation_sensitivity[layer_name]
    drop_at_2bit = layer_sens.get(2, 100.0)  # Default: assume critical
    drop_at_4bit = layer_sens.get(4, 100.0)
    drop_at_6bit = layer_sens.get(6, 100.0)
    
    # CRITICAL LAYER CHECK (>10% drop at A2)
    CRITICAL_THRESHOLD = 10.0
    is_critical = drop_at_2bit > CRITICAL_THRESHOLD
    
    if is_critical:
        return 8  # Always safe for critical layers
    
    # ROBUST LAYER: Apply mode-specific strategy
    if mode == 'safe':
        # Conservative matching: W2→A4 (don't use A2)
        if weight_bits >= 8:
            return 8
        elif weight_bits == 6:
            return 6
        elif weight_bits == 4:
            return 4
        elif weight_bits == 2:
            return 4  # Conservative: don't use A2 even for W2
        else:
            return 8
    
    elif mode == 'matched':
        # Exact match: W→A
        return weight_bits
    
    elif mode == 'aggressive':
        # Push to lowest possible based on sensitivity
        if drop_at_2bit < 2.0:  # Very robust
            return 2
        elif drop_at_4bit < 5.0:  # Moderately robust
            return 4
        elif drop_at_6bit < 8.0:  # Slightly sensitive
            return 6
        else:
            return 8
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'safe', 'matched', or 'aggressive'")


def generate_activation_config(weight_config, activation_sensitivity, mode='safe'):
    """
    Generate activation bit-width config from weight config.
    
    Args:
        weight_config: Dict {layer_name: weight_bits}
        activation_sensitivity: Dict {layer_name: {bits: sensitivity}}
        mode: 'safe', 'matched', or 'aggressive'
    
    Returns:
        dict: {layer_name: activation_bits}
    """
    activation_config = {}
    
    critical_count = 0
    robust_count = 0
    
    for layer_name, weight_bits in weight_config.items():
        # Skip granular configs (should only be layer-wise)
        if isinstance(weight_bits, list):
            print(f"  [WARNING] Skipping granular layer: {layer_name}")
            activation_config[layer_name] = 8  # Default safe
            continue
        
        activation_bits = get_activation_bits(layer_name, weight_bits, 
                                               activation_sensitivity, mode)
        activation_config[layer_name] = activation_bits
        
        # Track statistics
        if activation_bits == 8 and weight_bits < 8:
            critical_count += 1
        else:
            robust_count += 1
    
    return activation_config, critical_count, robust_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate mixed-precision activation configuration'
    )
    parser.add_argument('--weight-config', type=str, required=True,
                        help='Weight bit-width config JSON')
    parser.add_argument('--activation-sensitivity', type=str, required=True,
                        help='Activation sensitivity CSV')
    parser.add_argument('--output', type=str, required=True,
                        help='Output activation config JSON')
    parser.add_argument('--mode', type=str, default='safe',
                        choices=['safe', 'matched', 'aggressive'],
                        help='Matching mode (default: safe)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MIXED-PRECISION ACTIVATION CONFIG GENERATOR")
    print(f"{'='*60}")
    print(f"Weight config: {args.weight_config}")
    print(f"Activation sensitivity: {args.activation_sensitivity}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    # Load inputs
    print("Loading weight configuration...")
    with open(args.weight_config, 'r') as f:
        weight_config = json.load(f)
    print(f"  Loaded {len(weight_config)} layers")
    
    print("\nLoading activation sensitivity...")
    activation_sensitivity = load_activation_sensitivity(args.activation_sensitivity)
    print(f"  Loaded {len(activation_sensitivity)} layers")
    
    # Generate activation config
    print(f"\nGenerating activation config (mode={args.mode})...")
    activation_config, critical_count, robust_count = generate_activation_config(
        weight_config, activation_sensitivity, args.mode
    )
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total layers: {len(activation_config)}")
    print(f"  Critical (forced A8): {critical_count}")
    print(f"  Robust (matched): {robust_count}")
    
    # Bit distribution
    bit_dist = {}
    for bits in activation_config.values():
        bit_dist[bits] = bit_dist.get(bits, 0) + 1
    
    print(f"\nActivation bit-width distribution:")
    for bits in sorted(bit_dist.keys(), reverse=True):
        count = bit_dist[bits]
        pct = (count / len(activation_config)) * 100
        print(f"  A{bits}: {count:3d} layers ({pct:5.1f}%)")
    
    avg_bits = sum(b * c for b, c in bit_dist.items()) / len(activation_config)
    print(f"\nAverage activation bits: {avg_bits:.2f}")
    
    # Weight distribution for comparison
    weight_dist = {}
    for bits in weight_config.values():
        if isinstance(bits, list):
            bits = 8  # Granular → treat as 8
        weight_dist[bits] = weight_dist.get(bits, 0) + 1
    
    avg_weight_bits = sum(b * c for b, c in weight_dist.items()) / len(weight_config)
    print(f"Average weight bits:     {avg_weight_bits:.2f}")
    
    print(f"\nCombined: W{avg_weight_bits:.1f}/A{avg_bits:.1f}")
    
    # Computational savings estimate
    baseline_ops = len(activation_config) * 8 * 8  # W8/A8
    actual_ops = sum(
        weight_config.get(layer, 8) * activation_config[layer] 
        for layer in activation_config
    )
    savings = (1 - actual_ops / baseline_ops) * 100
    print(f"\nEstimated computational savings vs W8/A8: {savings:.1f}%")
    
    print(f"{'='*60}\n")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(activation_config, f, indent=2)
    
    print(f"Saved activation config to: {args.output}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEPS")
    print(f"{'='*60}")
    print(f"1. Validate PTQ accuracy:")
    print(f"   python validate_config.py \\")
    print(f"     --weight-config {args.weight_config} \\")
    print(f"     --activation-config {args.output}")
    print(f"\n2. Run QAT if needed:")
    print(f"   python qat_training.py \\")
    print(f"     --config {args.weight_config} \\")
    print(f"     --activation-config {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
