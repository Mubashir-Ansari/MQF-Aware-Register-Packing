"""
Joint W=A Greedy Search with Co-optimization Constraint
========================================================

Generates optimal mixed-precision configuration where EVERY layer
has matching weight and activation bit-widths (W=A constraint).

Algorithm:
  1. Load joint sensitivity profile (from joint_sensitivity.py)
  2. Initialize all layers to highest bit-width (e.g., W8/A8)
  3. Greedily reduce bit-widths of least sensitive layers
  4. CONSTRAINT: Always assign same bits to W and A in each layer
  5. Stop when target accuracy drop is reached

Usage:
    python joint_search.py \
        --model levit \
        --checkpoint models/best3_levit_model_cifar10.pth \
        --profile levit_joint_sensitivity.csv \
        --dataset cifar10 \
        --bits 2 4 6 8 \
        --target-drop 3.0

    # Outputs: levit_config_2_4_6_8.json (+ _weight.json, _activation.json)
"""

import argparse
import csv
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.hardware_sim import RegisterPackingSimulator
import models.alexnet
sys.modules['__main__'].fasion_mnist_alexnet = models.alexnet.AlexNet


def load_joint_sensitivity(profile_csv):
    """
    Load joint sensitivity profile from CSV.

    Returns:
        dict: {layer_name: {bits: sensitivity_score}}
    """
    sensitivity = {}

    with open(profile_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            sensitivity[layer] = {'baseline_acc': float(row['baseline_acc'])}

            # Extract sensitivity scores
            for key, value in row.items():
                if key.startswith('sensitivity_') and 'bit' in key:
                    bits = int(key.replace('sensitivity_', '').replace('bit', ''))
                    sensitivity[layer][bits] = float(value)

    return sensitivity

def get_filter_counts(model):
    """Get number of output channels/features for each quantizable layer."""
    counts = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            counts[name] = module.weight.shape[0]
    return counts

def get_layer_param_counts(model):
    """Get total parameter counts for each quantizable layer."""
    counts = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            counts[name] = module.weight.numel()
    return counts

def calculate_granular_packing(weights_shape, bit_widths, register_size=16):
    """
    Simulates granular packing savings for a weight tensor.
    bit_widths can be a single int or a list/tensor of ints per channel.
    """
    sim = RegisterPackingSimulator(register_size)
    num_channels = weights_shape[0]
    
    if isinstance(bit_widths, int):
        bit_widths = [bit_widths] * num_channels
    
    total_d = 0
    total_savings_fp32 = 0
    
    params_per_channel = 1
    for dim in weights_shape[1:]:
        params_per_channel *= dim
        
    for bw in bit_widths:
        d = sim.find_max_packing_factor(bw, bw) # W=A
        total_d += d
        sav_fp32, _ = sim.calculate_register_savings({'weight': bw}, params_per_channel)
        total_savings_fp32 += sav_fp32
        
    avg_d = total_d / num_channels
    return avg_d, total_savings_fp32


def hrp_greedy_search(
    sensitivity,
    bit_choices,
    target_drop=3.0,
    baseline_acc=None,
    register_size=16,
    filter_counts=None,
    param_counts=None,
    max_layer_budget_share=0.35,
    min_layers_to_modify=3
):
    """
    Heterogeneous Register Packing (HRP) Greedy Search.
    Optimizes for MAC Throughput (d) while staying under target accuracy drop.
    """
    if baseline_acc is None:
        sample_layer = next(iter(sensitivity.values()))
        baseline_acc = sample_layer.get('baseline_acc', 100.0)

    sim = RegisterPackingSimulator(register_size)
    
    # Sort bit choices (highest to lowest)
    bit_choices_sorted = sorted(bit_choices, reverse=True)
    max_bits = bit_choices_sorted[0]

    print("\n" + "="*70)
    print("HETEROGENEOUS REGISTER PACKING (HRP) SEARCH")
    print("="*70)
    print(f"Register Size: {register_size}-bit")
    print(f"Target Drop: {target_drop}%")
    print(f"Max Layer Budget Share: {max_layer_budget_share:.2f}")
    print(f"Min Layers To Modify: {min_layers_to_modify}")
    
    # Initialize all layers to max bits and optimal d
    config = {}
    for layer in sensitivity.keys():
        d = sim.find_max_packing_factor(max_bits, max_bits)
        config[layer] = {'w_bits': max_bits, 'a_bits': max_bits, 'd': d}

    # Tracking search state
    baseline_d = sim.find_max_packing_factor(max_bits, max_bits)
    baseline_total_d = len(config) * baseline_d
    
    # Pre-calculate all possible valid moves for each layer
    all_possible_moves = []
    for layer, scores in sensitivity.items():
        current_w, current_a = config[layer]['w_bits'], config[layer]['a_bits']
        current_d = config[layer]['d']
        
        for bits in bit_choices_sorted:
            if bits >= current_w: continue # Only lowering for now
            
            if bits in scores:
                drop = scores[bits]
                new_d = sim.find_max_packing_factor(bits, bits)
                d_gain = new_d - current_d
                layer_params = (param_counts or {}).get(layer, 1)
                # Register reduction proxy: params * (1/d_old - 1/d_new)
                reg_gain = layer_params * ((1.0 / max(current_d, 1e-9)) - (1.0 / max(new_d, 1e-9)))

                # Keep strict hardware-aware objective:
                # if packing factor and register reduction do not improve, do not spend accuracy budget.
                if d_gain > 0 and reg_gain > 0:
                    score = reg_gain / (drop + 0.01)
                    all_possible_moves.append({
                        'layer': layer,
                        'w_bits': bits,
                        'a_bits': bits,
                        'old_bits': current_w,
                        'd': new_d,
                        'd_gain': d_gain,
                        'reg_gain': reg_gain,
                        'drop': drop,
                        'score': score
                    })

    # Sort moves by score (highest efficiency first)
    all_possible_moves.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nEvaluating {len(all_possible_moves)} potential moves...")
    
    moves_made = []
    layers_modified = set()

    current_drop = 0.0
    layer_pool = list(all_possible_moves) # All possible moves
    
    per_layer_drop_cap = max_layer_budget_share * target_drop

    while layer_pool and current_drop < target_drop:
        # Find the best move from the remaining pool
        # This assumes layer_pool is already sorted by score, but we need to re-evaluate
        # if a layer has already been modified.
        
        best_move = None
        for i, move_candidate in enumerate(layer_pool):
            if move_candidate['layer'] not in layers_modified:
                best_move = move_candidate
                layer_pool.pop(i) # Remove from pool once considered
                break
        
        if not best_move:
            break # No more unique layers to modify
        
        layer = best_move['layer']
        
        # BUDGET CHECK for move
        remaining_budget = target_drop - current_drop
        allowed_drop = min(remaining_budget, per_layer_drop_cap)
        if allowed_drop <= 0:
            break

        if best_move['drop'] > allowed_drop:
            # Use granular split to satisfy both global budget and per-layer budget cap.
            fraction = allowed_drop / best_move['drop']

            # Conservative cap to reduce accuracy cliffs in critical/early layers.
            layer_name_lower = layer.lower()
            if "conv1" in layer_name_lower:
                max_fraction = 0.20
            elif "conv" in layer_name_lower:
                max_fraction = 0.30
            elif "fc" in layer_name_lower:
                max_fraction = 0.50
            else:
                max_fraction = 0.35

            if best_move.get('old_bits') == 8 and best_move['w_bits'] == 2:
                fraction = min(fraction, max_fraction)

            if fraction <= 0:
                continue

            best_move['is_granular'] = True
            best_move['fraction'] = fraction
            # Calculate bit distribution percentages
            # We assume splitting between old_bits and move['w_bits']
            old_bits = config[layer]['w_bits']
            new_bits = best_move['w_bits']
            
            best_move['granular_weights'] = {
                str(new_bits): round(fraction * 100, 1),
                str(old_bits): round((1 - fraction) * 100, 1)
            }
            best_move['drop'] = best_move['drop'] * fraction
            # Adjust packing factor average
            old_d = config[layer]['d']
            new_d = best_move['d']
            best_move['d'] = (fraction * new_d) + ((1-fraction) * old_d)

        # Apply move
        move = best_move
        layer = move['layer']
        current_drop += move['drop']
        layers_modified.add(layer)
        moves_made.append(move)
        
        config[layer]['w_bits'] = move['w_bits']
        config[layer]['a_bits'] = move['a_bits']
        config[layer]['d'] = move['d']
        
        print(f"[Move {len(moves_made)}] {layer:30s}: -> Mixed Precision (d={move['d']:.2f}) "
              f"[Drop: +{move['drop']:.2f}%, Total: {current_drop:.2f}%]")
        
        if move.get('is_granular', False):
            config[layer]['granular_dist'] = move['granular_weights']
            
            # Generate actual bit-width list if filter_counts available
            if filter_counts and layer in filter_counts:
                num_filters = filter_counts[layer]
                new_bits = move['w_bits']
                old_bits = move.get('old_bits', max_bits)
                fraction = move['fraction']
                
                num_new = int(round(fraction * num_filters))
                num_old = num_filters - num_new
                
                bit_list = [new_bits] * num_new + [old_bits] * num_old
                config[layer]['bit_list'] = bit_list
                
            print(f"      [GRANULAR] Distribution: {move['granular_weights']}")

    # Reformat for output
    final_config = {
        'config': {},
        'metadata': {
            'target_drop': target_drop,
            'final_drop': round(current_drop, 2),
            'register_size': register_size,
            'num_moves': len(moves_made)
        }
    }
    
    total_d = 0
    for layer, c in config.items():
        final_config['config'][layer] = {
            'weight': c.get('bit_list', c['w_bits']),
            'activation': c.get('bit_list', c['a_bits']),
            'packing_factor': round(c['d'], 2),
            'granular_dist': c.get('granular_dist', {str(c['w_bits']): 100.0})
        }
        total_d += c['d']
    
    avg_packing = total_d / len(config)
    baseline_avg_packing = baseline_d if baseline_d > 0 else 1.0
    final_config['metadata']['avg_packing_factor'] = round(avg_packing, 2)
    final_config['metadata']['total_throughput_gain'] = round(avg_packing / baseline_avg_packing, 2)

    # Parameter-weighted packing metrics (more representative than layer-average)
    if param_counts:
        total_params = sum(param_counts.get(layer, 0) for layer in config.keys())
        if total_params > 0:
            weighted_packing = sum(param_counts.get(layer, 0) * config[layer]['d'] for layer in config.keys()) / total_params
            final_config['metadata']['weighted_avg_packing_factor'] = round(weighted_packing, 2)
            final_config['metadata']['weighted_throughput_gain'] = round(weighted_packing / baseline_avg_packing, 2)
    
    # Calculate bit distribution
    bit_distribution = {}
    for c in config.values():
        b = c['w_bits']
        bit_distribution[b] = bit_distribution.get(b, 0) + 1
        
    avg_bits = sum(c['w_bits'] for c in config.values()) / len(config)

    stats = {
        'total_layers': len(config),
        'estimated_drop': round(current_drop, 2),
        'total_throughput_gain': (sum(c['d'] for c in config.values()) / baseline_total_d) if baseline_total_d > 0 else 1.0,
        'baseline_throughput': baseline_total_d,
        'avg_packing_factor': sum(c['d'] for c in config.values()) / len(config),
        'moves_made': len(moves_made),
        'bit_distribution': bit_distribution,
        'average_bits': round(avg_bits, 2),
        'baseline_accuracy': baseline_acc,
        'target_drop': target_drop,
        'estimated_accuracy': round(baseline_acc - current_drop, 2)
    }

    print(f"\n{'='*70}")
    print("HRP SEARCH COMPLETE")
    print(f"Total Throughput Gain: {stats['total_throughput_gain']}x (Avg d: {stats['avg_packing_factor']:.2f})")
    print(f"Estimated Accuracy Drop: {stats['estimated_drop']:.2f}%")
    if len(moves_made) < min_layers_to_modify:
        print(f"[WARN] Modified only {len(moves_made)} layers (< min_layers_to_modify={min_layers_to_modify}).")
    print("="*70)

    return final_config, stats


def save_config(config, output_path, stats):
    """
    Save configuration to JSON files.

    Generates TWO separate config files for compatibility with validate_config.py:
    1. *_weight_config.json - Weight bit-widths {layer: int}
    2. *_activation_config.json - Activation bit-widths {layer: int}

    Both files have IDENTICAL values (W=A constraint enforced).
    """
    # Extract weight and activation configs (they're identical due to W=A constraint)
    weight_config = {}
    activation_config = {}

    # Determine source config for flat files
    actual_config_dict = config.get('config', config) if isinstance(config, dict) else config
    
    for layer, details in actual_config_dict.items():
        # Handle case where details is a dict (hrp) or a single int
        if isinstance(details, dict):
            weight_config[layer] = details.get('weight', 8)
            activation_config[layer] = details.get('activation', 8)
        else:
            weight_config[layer] = details
            activation_config[layer] = details

    # Determine output paths
    # Remove .json extension and append _weight.json or _activation.json
    base_name = output_path.rsplit('.json', 1)[0]
    weight_path = f"{base_name}_weight.json"
    activation_path = f"{base_name}_activation.json"

    # Save weight config
    with open(weight_path, 'w') as f:
        json.dump(weight_config, f, indent=2)

    # Save activation config
    with open(activation_path, 'w') as f:
        json.dump(activation_config, f, indent=2)

    # Save the joint config for reference (Avoid double nesting)
    output_data = config if 'config' in config else {
        'config': actual_config_dict, # Use the extracted flat config
        'metadata': {
            'constraint': 'HRP (Heterogeneous Register Packing)',
            'total_layers': stats.get('total_layers', len(weight_config)),
            'bit_distribution': stats.get('bit_distribution', {}),
            'average_bits': stats.get('average_bits', 0),
            'estimated_drop': stats.get('estimated_drop', 0.0),
            'estimated_accuracy': stats.get('estimated_accuracy', 0),
            'baseline_accuracy': stats.get('baseline_accuracy', 0),
            'target_drop': stats.get('target_drop', 0),
            'total_throughput_gain': stats.get('total_throughput_gain', 0),
            'avg_packing_factor': stats.get('avg_packing_factor', 0)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Configuration saved:")
    print(f"  - Weight config:      {weight_path}")
    print(f"  - Activation config:  {activation_path}")
    print(f"  - Joint config (ref): {output_path}")
    print(f"✓ Format: {{layer: bits_int}} (compatible with validate_config.py)")
    print(f"✓ Constraint: W=A enforced (all {stats['total_layers']} layers have matching bits)")

    return weight_path, activation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint W=A Greedy Search with Co-optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-generates config filename)
  python joint_search.py \\
      --model levit \\
      --checkpoint models/best3_levit_model_cifar10.pth \\
      --profile levit_joint_sensitivity.csv \\
      --dataset cifar10

  # Custom target drop and output
  python joint_search.py \\
      --model vgg11_bn \\
      --checkpoint checkpoints/vgg11_bn.pt \\
      --profile vgg_joint_sensitivity.csv \\
      --dataset cifar10 \\
      --target-drop 2.0 \\
      --output results/vgg_joint_config.json
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--profile', type=str, required=True,
                       help='Joint sensitivity CSV file')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'gtsrb', 'fashionmnist'],
                       help='Dataset name')
    parser.add_argument('--register-size', type=int, default=16,
                       help='Hardware register size in bits (default: 16)')

    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output config JSON path (default: {model}_config_{bits}.json)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3, 4, 8],
                       help='Available bit-widths (default: 2 3 4 8)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                       help='Target accuracy drop %% (default: 3.0)')
    parser.add_argument('--baseline-acc', type=float, default=None,
                       help='Baseline accuracy (auto-detected if not provided)')
    parser.add_argument('--max-layer-budget-share', type=float, default=0.35,
                       help='Max fraction of target drop consumed by one layer (default: 0.35)')
    parser.add_argument('--min-layers-to-modify', type=int, default=3,
                       help='Minimum desired number of modified layers (default: 3)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided (include bit-widths)
    if args.output is None:
        bits_str = "_".join(map(str, sorted(args.bits)))
        args.output = f"{args.model}_config_{bits_str}.json"
        print(f"No output file specified, using: {args.output}")

    # Validate inputs
    if not os.path.exists(args.profile):
        print(f"Error: Profile not found: {args.profile}")
        exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)

    print("="*70)
    print("SETUP")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Profile: {args.profile}")
    print(f"Dataset: {args.dataset}")
    print(f"Bit choices: {args.bits}")
    print(f"Target drop: {args.target_drop}%")
    print("="*70)

    # Load sensitivity profile
    print(f"\nLoading joint sensitivity profile...")
    sensitivity = load_joint_sensitivity(args.profile)
    print(f"✓ Loaded sensitivity data for {len(sensitivity)} layers")

    # Get Filter Counts (Needed for Granular Bit-Lists)
    print(f"\nLoading model to get filter counts...")
    if args.dataset == 'cifar100': num_classes = 100
    elif args.dataset == 'gtsrb': num_classes = 43
    elif args.dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10
    
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    filter_counts = get_filter_counts(model)
    param_counts = get_layer_param_counts(model)
    print(f"✓ Found {len(filter_counts)} quantizable layers")

    # Run HRP search
    config, stats = hrp_greedy_search(
        sensitivity=sensitivity,
        bit_choices=args.bits,
        target_drop=args.target_drop,
        baseline_acc=args.baseline_acc,
        register_size=args.register_size,
        filter_counts=filter_counts,
        param_counts=param_counts,
        max_layer_budget_share=args.max_layer_budget_share,
        min_layers_to_modify=args.min_layers_to_modify
    )

    # Save configuration
    save_config(config, args.output, stats)

    # Get the generated config paths (already returned from save_config)
    base_name_display = args.output.rsplit('.json', 1)[0]
    weight_path_display = f"{base_name_display}_weight.json"
    activation_path_display = f"{base_name_display}_activation.json"

    # Only print NEXT STEPS if not running inside the auto-engine
    if not os.environ.get('HRP_AUTO_ENGINE'):
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}")
        print(f"1. Validate the config:")
        print(f"   python validate_config.py \\")
        print(f"       --model {args.model} \\")
        print(f"       --checkpoint {args.checkpoint} \\")
        print(f"       --config {weight_path_display} \\")
        print(f"       --activation-config {activation_path_display} \\")
        print(f"       --dataset {args.dataset}")
        print(f"\n2. If accuracy is good, you're done!")
        print(f"   If accuracy drops too much, run QAT:")
        print(f"   python qat_training.py \\")
        print(f"       --model {args.model} \\")
        print(f"       --checkpoint {args.checkpoint} \\")
        print(f"       --config {weight_path_display} \\")
        print(f"       --activation-config {activation_path_display} \\")
        print(f"       --dataset {args.dataset}")
        print(f"{'='*70}\n")
