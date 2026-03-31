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
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.hardware_sim import RegisterPackingSimulator
import models.alexnet
sys.modules['__main__'].fasion_mnist_alexnet = models.alexnet.AlexNet
analysis_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "analysis"
if str(analysis_dir) not in sys.path:
    sys.path.append(str(analysis_dir))
from cgrp import packing_score_delta


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

def _ensure_bit_list(config_entry, num_units):
    """Return an explicit per-unit bit list for iterative granular refinement."""
    if num_units <= 0:
        return []
    if 'bit_list' in config_entry and isinstance(config_entry['bit_list'], list):
        bit_list = [int(b) for b in config_entry['bit_list']]
        if len(bit_list) == num_units:
            return bit_list
    return [int(config_entry['w_bits'])] * num_units

def _avg_packing_from_bits(bit_list, sim):
    if not bit_list:
        return 1.0
    return sum(sim.find_max_packing_factor(int(b), int(b)) for b in bit_list) / len(bit_list)

def _granular_dist_from_bits(bit_list):
    counts = {}
    total = len(bit_list)
    for b in bit_list:
        counts[int(b)] = counts.get(int(b), 0) + 1
    return {str(b): round((counts[b] / total) * 100.0, 1) for b in sorted(counts.keys())}

def _replace_bits(bit_list, source_bit, target_bit, num_to_move):
    """Deterministically lower the first num_to_move units from source_bit to target_bit."""
    if num_to_move <= 0:
        return list(bit_list)
    out = list(bit_list)
    moved = 0
    for i, b in enumerate(out):
        if int(b) == int(source_bit):
            out[i] = int(target_bit)
            moved += 1
            if moved >= num_to_move:
                break
    return out

def _estimate_layer_priority(layer, param_counts, filter_counts):
    """Bias refinement toward layers with the largest hardware impact."""
    return float((param_counts or {}).get(layer, 0)) + 16.0 * float((filter_counts or {}).get(layer, 0))

def _estimate_storage_words_from_bits(bit_list, params_per_unit, register_size):
    """Approximate packed storage words for a layer from its current per-unit bit assignment."""
    if not bit_list or register_size <= 0:
        return 0.0
    return sum((params_per_unit * int(b)) / float(register_size) for b in bit_list)

def hrp_greedy_search(
    sensitivity,
    bit_choices,
    target_drop=3.0,
    baseline_acc=None,
    register_size=16,
    filter_counts=None,
    param_counts=None,
    max_layer_budget_share=0.35,
    min_layers_to_modify=3,
    refinement_passes=1,
    refinement_layer_budget_scale=1.75,
    min_high_impact_priority=0.0,
    storage_gain_weight=1.0,
    score_gamma=0.2
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

    # Track cumulative estimated drop per layer to keep iterative refinement bounded.
    moves_made = []
    current_drop = 0.0
    base_per_layer_drop_cap = max_layer_budget_share * target_drop
    layer_drop_spent = {layer: 0.0 for layer in sensitivity.keys()}
    layer_priority = {
        layer: _estimate_layer_priority(layer, param_counts, filter_counts)
        for layer in sensitivity.keys()
    }
    if min_high_impact_priority <= 0.0:
        min_high_impact_priority = max(layer_priority.values(), default=0.0) * 0.25

    # Co-optimization weights
    SCORE_ALPHA = 0.4
    SCORE_BETA = 0.4
    SCORE_GAMMA = score_gamma
    SCORE_DELTA = 1.0

    print(f"\n[METRIC] Multi-Pass Search initialized...")

    def collect_valid_moves(refinement_mode=False):
        valid_moves = []
        for layer, scores in list(sensitivity.items()):
            num_units = (filter_counts or {}).get(layer, 1)
            bit_list = _ensure_bit_list(config[layer], num_units)
            current_d = _avg_packing_from_bits(bit_list, sim)
            layer_params = (param_counts or {}).get(layer, 1)
            params_per_unit = layer_params / max(num_units, 1)
            current_storage_words = _estimate_storage_words_from_bits(bit_list, params_per_unit, register_size)
            
            # Constraint Enforcement
            tier = tier_map.get(layer, "Tier 3")
            min_bits = 2
            if "Tier 1" in tier: min_bits = 8
            if "Tier 2" in tier: min_bits = 4

            present_bits = sorted(set(int(b) for b in bit_list), reverse=True)
            for source_bit in present_bits:
                if source_bit <= min_bits:
                    continue
                eligible_units = sum(1 for b in bit_list if int(b) == source_bit)
                if eligible_units <= 0:
                    continue

                source_drop = scores.get(source_bit, 0.0)
                future_gain_exists = any(
                    lower in scores and lower < source_bit and sim.find_max_packing_factor(lower, lower) > sim.find_max_packing_factor(source_bit, source_bit)
                    for lower in bit_choices_sorted
                )

                for bits in bit_choices_sorted:
                    if bits >= source_bit:
                        continue
                    if bits < min_bits:
                        continue
                    if bits not in scores:
                        continue

                    drop_val = scores[bits]
                    full_marginal_drop = max(0.0, drop_val - source_drop)
                    moved_fraction_full = eligible_units / max(len(bit_list), 1)
                    predicted_drop = full_marginal_drop * moved_fraction_full

                    new_bit_list = _replace_bits(bit_list, source_bit, bits, eligible_units)
                    new_d = _avg_packing_from_bits(new_bit_list, sim)
                    d_gain = new_d - current_d
                    reg_gain = layer_params * ((1.0 / max(current_d, 1e-9)) - (1.0 / max(new_d, 1e-9)))
                    new_storage_words = _estimate_storage_words_from_bits(new_bit_list, params_per_unit, register_size)
                    storage_gain = max(0.0, current_storage_words - new_storage_words)
                    current_channels = list(zip(bit_list, bit_list))
                    bops_reduction_term = max(0.0, d_gain)
                    storage_reduction_term = max(0.0, storage_gain)
                    accuracy_proxy_term = predicted_drop
                    pack_delta = 0.0
                    if bit_list:
                        move_idx = next(
                            (idx for idx, value in enumerate(bit_list) if int(value) == int(source_bit)),
                            None
                        )
                        if move_idx is not None:
                            pack_delta = packing_score_delta(
                                channel_idx=move_idx,
                                candidate_bw=bits,
                                candidate_ba=bits,
                                current_channels=current_channels,
                                R=register_size,
                            )

                    unlock_bonus = 0.0
                    if d_gain <= 0 and future_gain_exists and full_marginal_drop <= 3.0:
                        unlock_bonus = 0.10 * layer_params

                    hardware_gain = reg_gain + (storage_gain_weight * storage_gain) + unlock_bonus
                    if reg_gain > 0 or storage_gain > 0 or unlock_bonus > 0:
                        score = (
                            SCORE_ALPHA * bops_reduction_term
                            + SCORE_BETA * storage_reduction_term
                            + SCORE_GAMMA * pack_delta
                            - SCORE_DELTA * accuracy_proxy_term
                        )
                        valid_moves.append({
                            'layer': layer,
                            'w_bits': bits,
                            'a_bits': bits,
                            'old_bits': source_bit,
                            'd': new_d,
                            'reg_gain': reg_gain,
                            'storage_gain': storage_gain,
                            'hardware_gain': hardware_gain,
                            'pack_delta': pack_delta,
                            'drop': predicted_drop,
                            'full_marginal_drop': full_marginal_drop,
                            'eligible_units': eligible_units,
                            'current_bit_list': bit_list,
                            'score': score,
                            'priority': layer_priority.get(layer, 0.0)
                        })
        if refinement_mode:
            high_impact_moves = [m for m in valid_moves if m['priority'] >= min_high_impact_priority]
            if high_impact_moves:
                return high_impact_moves
        return valid_moves

    def apply_best_move(candidate_pool, per_layer_drop_cap, refinement_mode=False):
        nonlocal current_drop
        if not candidate_pool:
            return False

        if not refinement_mode and len(set(m['layer'] for m in moves_made)) < min_layers_to_modify:
            seen_layers = {mm['layer'] for mm in moves_made}
            unseen_layer_moves = [m for m in candidate_pool if m['layer'] not in seen_layers]
            candidate_pool = unseen_layer_moves if unseen_layer_moves else candidate_pool

        candidate_pool.sort(key=lambda x: (x['score'], x['priority']), reverse=True)
        best_candidate = candidate_pool[0]
        layer = best_candidate['layer']
        
        remaining_budget = target_drop - current_drop
        remaining_layer_cap = max(0.0, per_layer_drop_cap - layer_drop_spent[layer])
        allowed_drop = min(remaining_budget, remaining_layer_cap)
        
        if best_candidate['drop'] > allowed_drop:
            if allowed_drop <= 0:
                return False
            
            fraction = allowed_drop / max(best_candidate['drop'], 1e-4)
            tier = tier_map.get(layer, "Tier 3")
            
            if "Tier 1" in tier: fraction = 0 
            elif "Tier 2" in tier: fraction = min(fraction, 0.45) 
            else: fraction = min(fraction, 0.95) 

            if fraction < 0.02: 
                return False

            best_candidate['is_granular'] = True
            best_candidate['fraction'] = fraction
            best_candidate['drop'] = allowed_drop
            units_to_move = max(1, int(round(best_candidate['eligible_units'] * fraction)))
            new_bit_list = _replace_bits(
                best_candidate['current_bit_list'],
                best_candidate['old_bits'],
                best_candidate['w_bits'],
                units_to_move
            )
            best_candidate['bit_list'] = new_bit_list
            best_candidate['d'] = _avg_packing_from_bits(new_bit_list, sim)
            best_candidate['granular_weights'] = _granular_dist_from_bits(new_bit_list)
        else:
            # Full move on all currently eligible units, still keeping the layer open for later refinement.
            new_bit_list = _replace_bits(
                best_candidate['current_bit_list'],
                best_candidate['old_bits'],
                best_candidate['w_bits'],
                best_candidate['eligible_units']
            )
            best_candidate['bit_list'] = new_bit_list
            best_candidate['d'] = _avg_packing_from_bits(new_bit_list, sim)
            best_candidate['granular_weights'] = _granular_dist_from_bits(new_bit_list)
            best_candidate['fraction'] = 1.0
            if all(int(b) == int(best_candidate['w_bits']) for b in new_bit_list):
                best_candidate['is_granular'] = False
            else:
                best_candidate['is_granular'] = True
        
        # Apply move
        current_drop += best_candidate['drop']
        layer_drop_spent[layer] += best_candidate['drop']
        moves_made.append(best_candidate)
        
        config[layer]['w_bits'] = best_candidate['w_bits']
        config[layer]['a_bits'] = best_candidate['a_bits']
        config[layer]['d'] = best_candidate['d']
        config[layer]['bit_list'] = best_candidate['bit_list']
        config[layer]['granular_dist'] = best_candidate['granular_weights']
        
        print(f"[Move {len(moves_made):2d}] {layer:15s} ({tier_map.get(layer, ''):15s}): "
              f"{best_candidate['old_bits']}b -> {best_candidate['w_bits']}b "
              f"(d={best_candidate['d']:.2f}) [Drop: +{best_candidate['drop']:.2f}%, Total: {current_drop:.2f}%]")
        
        if best_candidate.get('is_granular'):
            print(f"      [Surgical Dispatch] Split: {best_candidate['granular_weights']}")

        return True

    # Initial greedy pass.
    while current_drop < target_drop:
        valid_moves = collect_valid_moves(refinement_mode=False)
        if not valid_moves:
            break
        if not apply_best_move(valid_moves, base_per_layer_drop_cap, refinement_mode=False):
            break

    # Refinement pass: reuse remaining budget on high-impact layers with a relaxed per-layer cap.
    for refine_idx in range(max(0, refinement_passes)):
        if current_drop >= target_drop:
            break
        refined_any = False
        relaxed_cap = base_per_layer_drop_cap * refinement_layer_budget_scale
        print(f"[REFINE {refine_idx + 1}] Re-entering search with relaxed per-layer cap ({relaxed_cap:.2f}%).")
        while current_drop < target_drop:
            refinement_moves = collect_valid_moves(refinement_mode=True)
            if not refinement_moves:
                break
            if not apply_best_move(refinement_moves, relaxed_cap, refinement_mode=True):
                break
            refined_any = True
        if not refined_any:
            break

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
    parser.add_argument('--min-layers-to-modify', type=int, default=3)
    parser.add_argument('--refinement-passes', type=int, default=1)
    parser.add_argument('--refinement-layer-budget-scale', type=float, default=1.75)
    parser.add_argument('--min-high-impact-priority', type=float, default=0.0)
    parser.add_argument('--storage-gain-weight', type=float, default=1.0)
    parser.add_argument('--score-gamma', type=float, default=0.2)
    parser.add_argument('--baseline-acc', type=float, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.output is None:
        bits_tag = "_".join(str(b) for b in args.bits)
        args.output = f"{args.model}_config_{bits_tag}.json"

    sensitivity = load_joint_sensitivity(args.profile)
    num_classes = 100 if args.dataset == 'cifar100' else (43 if args.dataset == 'gtsrb' else 10)
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    filter_counts = get_filter_counts(model)
    param_counts = get_layer_param_counts(model)

    config, stats = hrp_greedy_search(
        sensitivity, args.bits, target_drop=args.target_drop,
        register_size=args.register_size, filter_counts=filter_counts,
        param_counts=param_counts, max_layer_budget_share=args.max_layer_budget_share,
        min_layers_to_modify=args.min_layers_to_modify,
        refinement_passes=args.refinement_passes,
        refinement_layer_budget_scale=args.refinement_layer_budget_scale,
        min_high_impact_priority=args.min_high_impact_priority,
        storage_gain_weight=args.storage_gain_weight,
        score_gamma=args.score_gamma,
        baseline_acc=args.baseline_acc
    )
    save_config(config, args.output, stats)
