"""
Joint W=A Greedy Search with Hybrid-Tier Operational Strategy
=============================================================

Incorporates surgical filter-level dispatch and three-tier logic:
1. Tier 1 (Sensitive): Locked 8-bit.
2. Tier 2 (Medium): 4-bit stability limit.
3. Tier 3 (Robust): Granular surgical MQF refinement.

Evolution of HRP:
- Allows multiple moves per layer (8 -> 4 -> 3 -> 2 recursive).
- Intelligence-driven bit-budgeting.
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
    """Load joint sensitivity profile from CSV."""
    sensitivity = {}
    with open(profile_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            sensitivity[layer] = {'baseline_acc': float(row['baseline_acc'])}
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
    Hybrid-Tier Surgical Search Pipeline.
    """
    if baseline_acc is None:
        sample_layer = next(iter(sensitivity.values()))
        baseline_acc = sample_layer.get('baseline_acc', 100.0)

    sim = RegisterPackingSimulator(register_size)
    bit_choices_sorted = sorted(bit_choices, reverse=True)
    max_bits = bit_choices_sorted[0]

    print("\n" + "="*70)
    print("HYBRID-TIER QUANTIZATION PIPELINE (HRP-MQF-H)")
    print("="*70)
    print(f"Operational Target: 16-bit Register Optimization")
    print(f"Target Budget:      {target_drop}% accuracy drop")
    print(f"Strategy:           Surgical Filter-Level Dispatch")
    print("="*70)
    
    # Initialize all layers to max bits
    config = {}
    for layer in sensitivity.keys():
        d = sim.find_max_packing_factor(max_bits, max_bits)
        config[layer] = {'w_bits': max_bits, 'a_bits': max_bits, 'd': d}

    baseline_d = sim.find_max_packing_factor(max_bits, max_bits)
    baseline_total_d = len(config) * baseline_d
    
    # --- Operational Tiers (Thesis Strategy) ---
    tier_map = {}
    for layer in sensitivity.keys():
        l_lower = layer.lower()
        if "conv1" in l_lower or "fc3" in l_lower:
            tier_map[layer] = "Tier 1 (Sensitive)"
        elif "conv" in l_lower:
            tier_map[layer] = "Tier 2 (Medium)"
        else:
            tier_map[layer] = "Tier 3 (Robust)"

    moves_made = []
    current_drop = 0.0
    per_layer_drop_cap = max_layer_budget_share * target_drop

    print(f"\n[METRIC] Multi-Pass Search initialized...")

    # MULTI-PASS GREEDY SEARCH
    while current_drop < target_drop:
        valid_moves = []
        for layer, scores in list(sensitivity.items()):
            current_w = config[layer]['w_bits']
            current_d = config[layer]['d']
            
            # Constraint Enforcement
            tier = tier_map.get(layer, "Tier 3")
            min_bits = 2
            if "Tier 1" in tier: min_bits = 8
            if "Tier 2" in tier: min_bits = 4

            for bits in bit_choices_sorted:
                if bits >= current_w: continue
                if bits < min_bits: continue
                
                if bits in scores:
                    drop_val = scores[bits]
                    # Marginal drop = Total(bits) - Total(current_w)
                    current_total_drop = scores.get(current_w, 0.0)
                    marginal_drop = max(0.0, drop_val - current_total_drop)
                    
                    new_d = sim.find_max_packing_factor(bits, bits)
                    d_gain = new_d - current_d
                    layer_params = (param_counts or {}).get(layer, 1)
                    reg_gain = layer_params * ((1.0 / max(current_d, 1e-9)) - (1.0 / max(new_d, 1e-9)))

                    if d_gain > 0 or (bits < current_w and marginal_drop < 0.2):
                        # Intelligence-driven Scoring (Reg gain per bit-drop cost)
                        score = reg_gain / (marginal_drop + 0.01)
                        valid_moves.append({
                            'layer': layer,
                            'w_bits': bits,
                            'a_bits': bits,
                            'old_bits': current_w,
                            'd': new_d,
                            'reg_gain': reg_gain,
                            'drop': marginal_drop,
                            'score': score
                        })
        
        if not valid_moves:
            break
            
        valid_moves.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = valid_moves[0]
        layer = best_candidate['layer']
        
        remaining_budget = target_drop - current_drop
        allowed_drop = min(remaining_budget, per_layer_drop_cap)
        
        if best_candidate['drop'] > allowed_drop:
            if allowed_drop <= 0: break
            
            fraction = allowed_drop / max(best_candidate['drop'], 1e-4)
            tier = tier_map.get(layer, "Tier 3")
            
            if "Tier 1" in tier: fraction = 0 
            elif "Tier 2" in tier: fraction = min(fraction, 0.45) 
            else: fraction = min(fraction, 0.95) 

            if fraction < 0.02: 
                sensitivity.pop(layer)
                continue

            best_candidate['is_granular'] = True
            best_candidate['fraction'] = fraction
            best_candidate['drop'] = allowed_drop
            new_d = best_candidate['d']
            old_d = config[layer]['d']
            best_candidate['d'] = (fraction * new_d) + ((1 - fraction) * old_d)
            
            best_candidate['granular_weights'] = {
                str(best_candidate['w_bits']): round(fraction * 100, 1),
                str(best_candidate['old_bits']): round((1 - fraction) * 100, 1)
            }
            sensitivity.pop(layer) # Surgical dispatch is always final move
        
        # Apply move
        current_drop += best_candidate['drop']
        moves_made.append(best_candidate)
        
        config[layer]['w_bits'] = best_candidate['w_bits']
        config[layer]['a_bits'] = best_candidate['a_bits']
        config[layer]['d'] = best_candidate['d']
        if best_candidate.get('is_granular'):
            config[layer]['granular_dist'] = best_candidate['granular_weights']
        
        print(f"[Move {len(moves_made):2d}] {layer:15s} ({tier_map.get(layer, ''):15s}): "
              f"{best_candidate['old_bits']}b -> {best_candidate['w_bits']}b "
              f"(d={best_candidate['d']:.2f}) [Drop: +{best_candidate['drop']:.2f}%, Total: {current_drop:.2f}%]")
        
        if best_candidate.get('is_granular'):
            print(f"      [Surgical Dispatch] Split: {best_candidate['granular_weights']}")
            if filter_counts and layer in filter_counts:
                num = filter_counts[layer]
                n_new = int(round(best_candidate['fraction'] * num))
                config[layer]['bit_list'] = [best_candidate['w_bits']] * n_new + [best_candidate['old_bits']] * (num - n_new)

    # Final stats
    final_config = {'config': {}, 'metadata': {
        'target_drop': target_drop, 'final_drop': round(current_drop, 2),
        'register_size': register_size, 'num_moves': len(moves_made),
        'pipeline': 'Hybrid-Tier-MQF'
    }}
    
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
    final_config['metadata']['avg_packing_factor'] = round(avg_packing, 2)
    final_config['metadata']['total_throughput_gain'] = round(avg_packing / baseline_d, 2)

    if param_counts:
        weighted_packing = sum(param_counts.get(layer, 0) * config[layer]['d'] for layer in config.keys()) / sum(param_counts.values())
        final_config['metadata']['weighted_avg_packing_factor'] = round(weighted_packing, 2)
        final_config['metadata']['weighted_throughput_gain'] = round(weighted_packing / baseline_d, 2)
    
    avg_bits = sum(c['w_bits'] for c in config.values()) / len(config)
    
    stats = {
        'total_layers': len(config),
        'estimated_drop': round(current_drop, 2),
        'avg_packing_factor': avg_packing,
        'average_bits': round(avg_bits, 2),
        'total_throughput_gain': final_config['metadata']['total_throughput_gain'],
        'baseline_accuracy': baseline_acc
    }

    print(f"\n{'='*70}")
    print("HYBRID-TIER SEARCH COMPLETE")
    print(f"Throughput Gain: {final_config['metadata']['total_throughput_gain']}x")
    print(f"Efficiency:      {int(avg_packing*100/sim.find_max_packing_factor(2,2)) if sim.find_max_packing_factor(2,2)>0 else 0}% theoretical max")
    print("="*70)

    return final_config, stats

def save_config(config, output_path, stats):
    weight_config = {}
    activation_config = {}
    for layer, details in config['config'].items():
        weight_config[layer] = details['weight']
        activation_config[layer] = details['activation']

    base_name = output_path.rsplit('.json', 1)[0]
    with open(f"{base_name}_weight.json", 'w') as f: json.dump(weight_config, f, indent=2)
    with open(f"{base_name}_activation.json", 'w') as f: json.dump(activation_config, f, indent=2)
    with open(output_path, 'w') as f: json.dump(config, f, indent=2)

    return f"{base_name}_weight.json", f"{base_name}_activation.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--profile', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--register-size', type=int, default=16)
    parser.add_argument('--target-drop', type=float, default=3.0)
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3, 4, 8])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-layer-budget-share', type=float, default=0.35)
    parser.add_argument('--baseline-acc', type=float, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}_config_hybrid.json"

    sensitivity = load_joint_sensitivity(args.profile)
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    filter_counts = get_filter_counts(model)
    param_counts = get_layer_param_counts(model)

    config, stats = hrp_greedy_search(
        sensitivity, args.bits, target_drop=args.target_drop,
        register_size=args.register_size, filter_counts=filter_counts,
        param_counts=param_counts, max_layer_budget_share=args.max_layer_budget_share,
        baseline_acc=args.baseline_acc
    )
    save_config(config, args.output, stats)
