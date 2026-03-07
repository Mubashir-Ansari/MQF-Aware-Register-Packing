"""
Auto-Quantize Engine with Joint W=A Co-optimization
====================================================

Updated version that enforces W=A constraint throughout the pipeline.

Key changes from original auto_quantize_engine.py:
1. Uses joint_sensitivity.py instead of layer_sensitivity.py
2. Uses joint_search.py instead of hardware_aware_search.py
3. Enforces W=A constraint (all layers have matching weight/activation bits)
4. BOPs calculation uses actual activation bits (not fixed 8-bit)

Usage:
    python auto_quantize_engine_joint.py \
        --model levit \
        --checkpoint models/best3_levit_model_cifar10.pth \
        --dataset cifar10 \
        --bits 4 6 8 \
        --target-drop 2.0
"""

import argparse
import os
import sys
import subprocess
import json
import time
import torch
import csv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pipeline import get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, get_fashionmnist_dataloader, evaluate_accuracy

# Fix for checkpoint loading (fasion_mnist_alexnet class mismatch)
import models.alexnet
import sys
fasion_mnist_alexnet = models.alexnet.AlexNet
sys.modules['__main__'].fasion_mnist_alexnet = models.alexnet.AlexNet
from models.model_loaders import load_model


def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"\n[ENGINE] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with return code {ret}: {cmd}")


def profile_needs_regeneration(profile_csv, current_baseline_acc, baseline_tol=5.0):
    """
    Detect stale/corrupt sensitivity profile.

    Regenerate if:
    1) File missing/empty
    2) Baseline in CSV differs too much from current measured baseline
    3) All/near-all sensitivity entries are zero (common bad-profile symptom)
    """
    if not os.path.exists(profile_csv):
        return True, "profile_missing"

    rows = []
    try:
        with open(profile_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return True, "profile_unreadable"

    if not rows:
        return True, "profile_empty"

    # Check baseline consistency
    csv_baselines = []
    sensitivity_vals = []
    for row in rows:
        try:
            csv_baselines.append(float(row.get("baseline_acc", 0.0)))
        except Exception:
            pass
        for k, v in row.items():
            if k.startswith("sensitivity_"):
                try:
                    sensitivity_vals.append(abs(float(v)))
                except Exception:
                    pass

    if not csv_baselines:
        return True, "baseline_missing"

    csv_baseline_avg = sum(csv_baselines) / len(csv_baselines)
    if abs(csv_baseline_avg - current_baseline_acc) > baseline_tol:
        return True, f"baseline_mismatch(csv={csv_baseline_avg:.2f}, current={current_baseline_acc:.2f})"

    if sensitivity_vals:
        max_sens = max(sensitivity_vals)
        if max_sens < 1e-6:
            return True, "all_zero_sensitivity"

    return False, "profile_ok"


def calculate_bops_joint(model, config, input_size=32):
    """
    Calculate Bit Operations (BOPs) for joint W=A mixed-precision.

    BOPs = sum over layers of: MACs × weight_bits × activation_bits

    Args:
        model: PyTorch model
        config: Joint W=A config (e.g., {"layer": {"weight": 4, "activation": 4}})
        input_size: Input image size for spatial dimension calculation

    Returns:
        BOPs in GigaBOPs (GBOPs)
    """
    import torch.nn as nn
    import numpy as np

    total_bops = 0

    for name, module in model.named_modules():
        if name in config:
            # Handle dict format: {"weight": bits, "activation": bits}
            if isinstance(config[name], dict):
                w_bits = config[name].get('weight', 8)
                a_bits = config[name].get('activation', 8)
            else:
                # Fallback: single integer (shouldn't happen with joint config)
                w_bits = config[name]
                a_bits = config[name]

            # Convert lists to average for BOPs calculation if granular
            def get_avg_bits(b):
                return sum(b)/len(b) if isinstance(b, list) else b
            
            wb = get_avg_bits(w_bits)
            ab = get_avg_bits(a_bits)

            if isinstance(module, nn.Conv2d):
                macs = module.in_channels * module.out_channels * \
                       module.kernel_size[0] * module.kernel_size[1]
                # Adjust for spatial output (this is an approximation if input_size is global)
                stride = module.stride[0] if hasattr(module.stride, '__getitem__') else module.stride
                # Simple spatial estimation
                h_out = input_size // stride
                w_out = h_out
                total_bops += (macs * h_out * w_out) * wb * ab

            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
                total_bops += macs * wb * ab

    return total_bops / 1e9  # GBOPs


def verify_wa_constraint(config):
    """
    Verify that all layers in config satisfy W=A constraint.

    Returns:
        (bool, int, int): (is_compliant, matching_layers, total_layers)
    """
    matching = 0
    total = 0

    # Handle both formats
    if 'config' in config:
        config = config['config']

    for layer_name, layer_config in config.items():
        if isinstance(layer_config, dict):
            w_bits = layer_config.get('weight', None)
            a_bits = layer_config.get('activation', None)

            if w_bits is not None and a_bits is not None:
                total += 1
                if w_bits == a_bits:
                    matching += 1

    is_compliant = (matching == total) if total > 0 else False
    return is_compliant, matching, total


def save_metrics(model_name, baseline_acc, ptq_acc, final_acc,
                 config_path, output_file,
                 used_qat=False, qat_threshold=2.0,
                 baseline_size_mb=None, quantized_size_mb=None,
                 probe_time=None, search_time=None, validation_time=None,
                 qat_time=None, compress_time=None, total_time=None,
                 baseline_bops=None, quantized_bops=None, bops_reduction=None,
                 wa_compliance=None, avg_bits=None):
    """Save comprehensive metrics to JSON"""
    from datetime import datetime

    metrics = {
        'model': model_name,
        'baseline_accuracy': round(baseline_acc, 2),
        'ptq_accuracy': round(ptq_acc, 2),
        'final_accuracy': round(final_acc, 2),
        'accuracy_drop': round(baseline_acc - final_acc, 2),
        'ptq_drop': round(baseline_acc - ptq_acc, 2),
        'baseline_size_mb': round(baseline_size_mb, 2) if baseline_size_mb else None,
        'quantized_size_mb': round(quantized_size_mb, 2) if quantized_size_mb else None,
        'compression_ratio': round(baseline_size_mb / quantized_size_mb, 2) if (baseline_size_mb and quantized_size_mb) else None,
        'bops_analysis': {
            'baseline_bops_gbops': round(baseline_bops, 2) if baseline_bops else None,
            'quantized_bops_gbops': round(quantized_bops, 2) if quantized_bops else None,
            'bops_reduction': round(bops_reduction, 2) if bops_reduction else None,
            'note': 'Calculated with joint W=A mixed-precision (variable activation bits)'
        },
        'joint_wa_optimization': {
            'constraint': 'HRP (Heterogeneous Register Packing)',
            'compliance': f"{wa_compliance['matching']}/{wa_compliance['total']}" if wa_compliance else None,
            'compliance_rate': round(wa_compliance['matching'] / wa_compliance['total'] * 100, 1) if wa_compliance else None,
            'average_bits': round(avg_bits, 2) if avg_bits else None,
            'avg_packing_factor': round(avg_packing_factor, 2) if 'avg_packing_factor' in locals() else None,
            'total_throughput_gain': round(total_throughput_gain, 2) if 'total_throughput_gain' in locals() else None
        },
        'config_file': config_path,
        'quantization_method': 'QAT' if used_qat else 'PTQ',
        'qat_triggered': used_qat,
        'qat_threshold': qat_threshold,
        'timing': {
            'probe_seconds': round(probe_time, 2) if probe_time is not None else None,
            'search_seconds': round(search_time, 2) if search_time is not None else None,
            'validation_seconds': round(validation_time, 2) if validation_time is not None else None,
            'qat_seconds': round(qat_time, 2) if qat_time is not None else None,
            'compress_seconds': round(compress_time, 2) if compress_time is not None else None,
            'total_seconds': round(total_time, 2) if total_time is not None else None,
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[METRICS] Saved to {output_file}")
    print(f"  Baseline: {baseline_acc:.2f}% | Final: {final_acc:.2f}% | Drop: {baseline_acc - final_acc:.2f}%")
    if baseline_bops and quantized_bops:
        print(f"  BOPs: {baseline_bops:.2f} GBOPs → {quantized_bops:.2f} GBOPs ({bops_reduction:.2f}x reduction)")
    if wa_compliance:
        print(f"  W=A Compliance: {wa_compliance['matching']}/{wa_compliance['total']} layers ({wa_compliance['matching']/wa_compliance['total']*100:.1f}%)")
    if 'avg_packing_factor' in metrics['joint_wa_optimization'] and metrics['joint_wa_optimization']['avg_packing_factor'] is not None:
        print(f"  MAC Throughput Gain: {metrics['joint_wa_optimization']['avg_packing_factor']:.2f}x (Avg d)")


def auto_quantize_joint(args):
    model_name = args.model
    checkpoint_path = args.checkpoint
    dataset = args.dataset
    target_drop = args.target_drop
    bit_choices = args.bits
    output_metrics = getattr(args, 'output_metrics', 'metrics.json')
    qat_threshold = args.qat_threshold
    gtsrb_use_train_val = getattr(args, 'gtsrb_use_train_val', False)
    gtsrb_val_ratio = getattr(args, 'gtsrb_val_ratio', 0.2)
    gtsrb_seed = getattr(args, 'gtsrb_seed', 42)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    eval_batch_size = getattr(args, 'batch_size', 128)
    if model_name == 'alexnet' and eval_batch_size > 32:
        eval_batch_size = 32
    """
    Auto-Quantize Engine with Joint W=A Co-optimization.

    Flow: Joint Probe -> Joint Search -> Validate (Gate) -> [QAT if needed]

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name (cifar10, cifar100, gtsrb)
        target_drop: Target accuracy drop for search (default: 3.0%)
        bit_choices: List of bit-widths to consider (default: [4, 6, 8])
        output_metrics: Metrics output file path (default: 'metrics.json')
        qat_threshold: Accuracy drop threshold to trigger QAT (default: 2.0%)
        gtsrb_use_train_val: Use internal Train folder split for GTSRB (default: False)
        gtsrb_val_ratio: Validation ratio for GTSRB internal split (default: 0.2)
        gtsrb_seed: Random seed for GTSRB internal split (default: 42)
    """
    print("="*70)
    print(f"AUTO-QUANTIZATION ENGINE (JOINT W=A CO-OPTIMIZATION)")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Bit-Choices: {bit_choices}")
    print(f"Constraint: W=A enforced (co-optimized)")
    print(f"Target Drop: {target_drop}%")
    print(f"QAT Threshold: {qat_threshold}%")
    print(f"Register Size: {getattr(args, 'register_size', 16)}-bit")
    if dataset == 'gtsrb' and gtsrb_use_train_val:
        print(f"GTSRB Mode: Internal Train/Val Split (val_ratio={gtsrb_val_ratio}, seed={gtsrb_seed})")
    print("="*70)

    # Initialize timing
    overall_start = time.time()
    probe_time = search_time = validation_time = qat_time = compress_time = 0.0
    
    # 0. Load Model & Measure Baseline (Always needed)
    print(f"\n[ENGINE] Loading {model_name} for baseline measurement...")
    # (FashionMNIST AlexNet fix is at top of file)
    if dataset == 'cifar100': num_classes = 100
    elif dataset == 'gtsrb': num_classes = 43
    elif dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10
    
    # Use already imported load_model
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    model.to(device)
    
    # Determine input size
    if model_name == 'alexnet':
        input_size = 227
    elif model_name in ['levit', 'swin']:
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:
        input_size = 224
        
    loader = get_fashionmnist_dataloader(batch_size=eval_batch_size, train=False, input_size=input_size)
    # Use already imported evaluate_accuracy
    acc_baseline = evaluate_accuracy(model, loader, device=device, max_samples=args.max_samples)
    print(f"Baseline Accuracy ({model_name} on {dataset}): {acc_baseline:.2f}%")

    # Paths (include bit-widths to avoid confusion between different runs)
    bits_str_filename = "_".join(map(str, sorted(bit_choices)))
    joint_profile_csv = f"{model_name}_sensitivity_{bits_str_filename}.csv"
    joint_config_json = f"{model_name}_config_{bits_str_filename}.json"
    weight_config_json = f"{model_name}_config_{bits_str_filename}_weight.json"
    activation_config_json = f"{model_name}_config_{bits_str_filename}_activation.json"

    # ---------------------------------------------------------
    # STEP 1: JOINT PROBE (Joint W=A Sensitivity Analysis)
    # ---------------------------------------------------------
    probe_start = time.time()
    regen, reason = profile_needs_regeneration(joint_profile_csv, acc_baseline)
    if regen:
        print(f"\n[STEP 1] Generating Joint W=A Sensitivity Profile ({joint_profile_csv})...")
        if reason != "profile_missing":
            print(f"[STEP 1] Rebuilding profile due to: {reason}")
        bits_str = " ".join(map(str, bit_choices))
        cmd = f"python quantization_framework/experiments/joint_sensitivity.py " \
              f"--model {model_name} " \
              f"--checkpoint {checkpoint_path} " \
              f"--dataset {dataset} " \
              f"--bits {bits_str} " \
              f"--device {device} " \
              f"--max-samples {args.max_samples}"
        # Note: joint_sensitivity.py auto-generates output filename
        run_command(cmd)
    else:
        print(f"\n[STEP 1] Found existing joint profile: {joint_profile_csv}")
    probe_time = time.time() - probe_start
    print(f"[TIMING] Joint profiling completed in {probe_time:.2f}s ({probe_time/60:.1f} min)")

    # ---------------------------------------------------------
    # STEP 2: JOINT SEARCH (Generate W=A Config)
    # ---------------------------------------------------------
    search_start = time.time()
    print(f"\n[STEP 2] Searching for Optimal W=A Configuration...")
    bits_str = " ".join(map(str, bit_choices))
    cmd = f"python quantization_framework/experiments/joint_search.py " \
          f"--model {model_name} " \
          f"--checkpoint {checkpoint_path} " \
          f"--profile {joint_profile_csv} " \
          f"--dataset {dataset} " \
          f"--bits {bits_str} " \
          f"--target-drop {target_drop} " \
          f"--register-size {getattr(args, 'register_size', 16)} " \
          f"--device {device} " \
          f"--baseline-acc {acc_baseline}"
    # Note: joint_search.py auto-generates output filename
    run_command(cmd)
    search_time = time.time() - search_start
    print(f"[TIMING] Joint search completed in {search_time:.2f}s ({search_time/60:.1f} min)")

    # ---------------------------------------------------------
    # STEP 2.5: VERIFY W=A CONSTRAINT
    # ---------------------------------------------------------
    print(f"\n[STEP 2.5] Verifying W=A Constraint Compliance...")
    cmd = f"python quantization_framework/experiments/verify_wa_constraint.py --config {joint_config_json}"
    try:
        run_command(cmd)
        print("[✓] W=A constraint verification passed!")
    except RuntimeError as e:
        print(f"[✗] WARNING: W=A constraint verification failed!")
        print(f"    This should not happen with joint_search.py. Check config manually.")

    # ---------------------------------------------------------
    # STEP 3: GATE (Validate PTQ)
    # ---------------------------------------------------------
    validation_start = time.time()
    print(f"\n[STEP 3] Validating PTQ Accuracy (The Gate)...")

    # Validate via subprocess using separate weight and activation configs
    cmd = f"python quantization_framework/experiments/validate_config.py " \
          f"--model {model_name} " \
          f"--checkpoint {checkpoint_path} " \
          f"--config {weight_config_json} " \
          f"--activation-config {activation_config_json} " \
          f"--dataset {dataset} " \
          f"--device {device} " \
          f"--batch-size {eval_batch_size} " \
          f"--max-samples {args.max_samples}"
    try:
        run_command(cmd)
    except RuntimeError as e:
        # On crowded GPUs, fallback to CPU validation instead of aborting the full pipeline.
        if device == 'cuda':
            print("[WARNING] CUDA validation failed (likely OOM). Retrying validation on CPU...")
            cmd_cpu = f"python quantization_framework/experiments/validate_config.py " \
                      f"--model {model_name} " \
                      f"--checkpoint {checkpoint_path} " \
                      f"--config {weight_config_json} " \
                      f"--activation-config {activation_config_json} " \
                      f"--dataset {dataset} " \
                      f"--device cpu " \
                      f"--batch-size 16 " \
                      f"--max-samples {args.max_samples}"
            run_command(cmd_cpu)
        else:
            raise e

    # Check Accuracy Internally
    print("[ENGINE] Internal validation check...")

    # Load all configs
    with open(weight_config_json, 'r') as f:
        weight_config = json.load(f)
    with open(activation_config_json, 'r') as f:
        activation_config = json.load(f)
    with open(joint_config_json, 'r') as f:
        joint_config_data = json.load(f)
        hrp_metadata = joint_config_data.get('metadata', {})
        avg_packing_factor = hrp_metadata.get('avg_packing_factor', 1.0)
        total_throughput_gain = hrp_metadata.get('total_throughput_gain', 1.0)
        weighted_avg_packing_factor = hrp_metadata.get('weighted_avg_packing_factor', None)
        weighted_throughput_gain = hrp_metadata.get('weighted_throughput_gain', None)

    # Verify W=A constraint by comparing weight and activation configs
    matching = 0
    total = 0
    for layer in weight_config.keys():
        if layer in activation_config:
            total += 1
            if weight_config[layer] == activation_config[layer]:
                matching += 1

    is_compliant = (matching == total) if total > 0 else False
    wa_compliance = {'matching': matching, 'total': total}

    if not is_compliant:
        print(f"[WARNING] W=A constraint not fully satisfied: {matching}/{total} layers match")
    else:
        print(f"[✓] W=A constraint satisfied: {matching}/{total} layers (100%)")

    # Determine number of classes based on dataset
    if dataset == 'cifar100': num_classes = 100
    elif dataset == 'gtsrb': num_classes = 43
    elif dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10

    # Load model
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    
    # ---------------------------------------------------------
    # STEP 3.1: GENERATE ALGO REPORT: REGISTER-MISMATCH ANALYSIS
    # ---------------------------------------------------------
    print("\n[STEP 3.1] Generating Hardware-Aware Registration Report...")
    
    # Build a param count map for register calculations
    param_counts = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            param_counts[name] = module.weight.numel()

    print("\n[ALGO REPORT: REGISTER-MISMATCH ANALYSIS]")
    print(f"{'Layer':12} | {'2b %':7} | {'4b %':7} | {'8b %':7} | {'Pack (A)':8} | {'Ew':5} | {'Status'}")
    print("-" * 92)
    
    # Fair comparison: Baseline (8-bit) should also be subjected to HRP constraints
    from quantization.hardware_sim import RegisterPackingSimulator
    from quantization.packing import ReQAPPackingPlanner
    reg_size = hrp_metadata.get('register_size', 16)
    sim = RegisterPackingSimulator(reg_size)
    planner = ReQAPPackingPlanner(register_size=reg_size, max_d=8)
    d_base = sim.find_max_packing_factor(8, 8) # Baseline is 8-bit Weight/Activation
    
    total_baseline_registers = 0
    total_mqf_registers = 0
    total_baseline_storage_words = 0
    total_mqf_storage_words = 0
    
    for layer, c in joint_config_data['config'].items():
        dist = c.get('granular_dist', {})
        p2 = dist.get('2', 0.0)
        p4 = dist.get('4', 0.0)
        p8 = dist.get('8', 0.0)

        w_bit = c.get('weight', 8)
        if not dist:
            if isinstance(w_bit, list):
                total = len(w_bit)
                if total > 0:
                    p2 = 100.0 * sum(1 for b in w_bit if b == 2) / total
                    p4 = 100.0 * sum(1 for b in w_bit if b == 4) / total
                    p8 = 100.0 * sum(1 for b in w_bit if b == 8) / total
            else:
                p2 = 100.0 if w_bit == 2 else 0.0
                p4 = 100.0 if w_bit == 4 else 0.0
                p8 = 100.0 if w_bit == 8 else 0.0

        pack_a = c.get('packing_factor', 1.0)
        status = "GRANULAR" if isinstance(w_bit, list) or len(dist) > 1 else "UNIFORM"

        if isinstance(w_bit, list) and len(w_bit) > 0:
            eff_w_bits = int(round(sum(w_bit) / len(w_bit)))
        elif dist:
            eff_w_bits = int(round(sum(int(k) * (v / 100.0) for k, v in dist.items() if str(k).isdigit())))
        else:
            eff_w_bits = int(w_bit) if isinstance(w_bit, int) else 8
        plan = planner.plan(eff_w_bits, eff_w_bits)

        print(f"{layer:12} | {p2:>6}% | {p4:>6}% | {p8:>6}% | {pack_a:<8.2f} | {plan.empty_bits_weight:<5d} | {status}")
        
        params = param_counts.get(layer, 0)
        
        # d_base is the FAIR packing for 8-bit baseline on the same hardware
        base_regs = params / d_base

        # Exact granular accounting: regs = sum(params_frac / d_frac)
        if isinstance(w_bit, list) and len(w_bit) > 0:
            from collections import Counter
            bit_counts = Counter(w_bit)
            mqf_regs = 0.0
            for bits, cnt in bit_counts.items():
                d_sub = sim.find_max_packing_factor(int(bits), int(bits))
                mqf_regs += cnt / max(d_sub, 1)
        elif dist:
            mqf_regs = 0.0
            for bits_s, pct in dist.items():
                try:
                    bits_i = int(bits_s)
                except ValueError:
                    continue
                d_sub = sim.find_max_packing_factor(bits_i, bits_i)
                mqf_regs += (params * (pct / 100.0)) / max(d_sub, 1)
        else:
            mqf_regs = params / pack_a if pack_a > 0 else base_regs
        
        total_baseline_registers += base_regs
        total_mqf_registers += mqf_regs

        # Raw storage packing perspective (without accumulation safety bound):
        # number of R-bit words needed for weights.
        # Baseline is fixed 8-bit weights.
        total_baseline_storage_words += (params * 8.0) / reg_size
        if isinstance(w_bit, list) and len(w_bit) > 0:
            total_mqf_storage_words += (sum(w_bit) / reg_size) * (params / len(w_bit))
        elif dist:
            weighted_bits = 0.0
            for bits_s, pct in dist.items():
                try:
                    bits_i = int(bits_s)
                except ValueError:
                    continue
                weighted_bits += bits_i * (pct / 100.0)
            total_mqf_storage_words += (params * weighted_bits) / reg_size
        else:
            bits_scalar = w_bit if isinstance(w_bit, int) else 8
            total_mqf_storage_words += (params * bits_scalar) / reg_size

    print("-" * 92)
    print(f"  Register Count (8-bit Baseline - HRP-Aware): {int(total_baseline_registers):,}")
    print(f"  MQF registers count:                         {int(total_mqf_registers):,}")
    
    storage_efficiency = (total_baseline_registers / total_mqf_registers * 100) if total_mqf_registers > 0 else 0
    # For user report, we want to show SAVINGS
    savings = (1.0 - (total_mqf_registers / total_baseline_registers)) * 100 if total_baseline_registers > 0 else 0
    print(f"  Estimated Register Savings:                  {savings:.2f}%")
    storage_savings = (1.0 - (total_mqf_storage_words / total_baseline_storage_words)) * 100 if total_baseline_storage_words > 0 else 0
    print(f"  Storage Words (R={reg_size}b, baseline 8b): {int(total_baseline_storage_words):,}")
    print(f"  MQF storage words (packed by bit-width):     {int(total_mqf_storage_words):,}")
    print(f"  Estimated Storage Savings:                   {storage_savings:.2f}%")
    base_plan = planner.plan(8, 8, d=d_base)
    print(f"  Packed ops supported:                        {', '.join(base_plan.ops_supported)}")
    if weighted_avg_packing_factor is not None:
        print(f"  Weighted Avg Packing (param-weighted):       {weighted_avg_packing_factor:.2f}")
    if weighted_throughput_gain is not None:
        print(f"  Weighted Throughput Gain:                    {weighted_throughput_gain:.2f}x")
    print(f"  Quantization Mode:                           JOINT W=A")
    print("-" * 70)

    # Remove redundant device detection later down
    print(f"[ENGINE] Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Determine input size
    if model_name == 'alexnet':
        input_size = 227
    elif model_name in ['levit', 'swin']:
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:  # fashionmnist, cifar10, cifar100, gtsrb
        input_size = 224

    # Load dataloader
    if dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=eval_batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(
            batch_size=eval_batch_size,
            train=False,
            input_size=input_size,
            use_train_val_split=gtsrb_use_train_val,
            val_ratio=gtsrb_val_ratio,
            seed=gtsrb_seed
        )
    elif dataset == 'fashionmnist':
        loader = get_fashionmnist_dataloader(batch_size=eval_batch_size, train=False, input_size=input_size)
    else:  # cifar10
        loader = get_cifar10_dataloader(batch_size=eval_batch_size, train=False, input_size=input_size)

    # Quick Baseline
    acc_baseline = evaluate_accuracy(model, loader, device=device, max_samples=args.max_samples)

    # Apply Config
    from experiments.validate_config import apply_mixed_precision, calibrate_activation_quantizers

    # Configs already loaded in validation check above - reuse them
    model, quantizers = apply_mixed_precision(model, weight_config,
                         quantize_weights=True,
                         quantize_activations=True,
                         act_bit_width=8,
                         activation_config=activation_config)

    # Calibrate activation quantizers
    if quantizers:
        calibrate_activation_quantizers(model, quantizers, loader, device=device, num_batches=10)

    acc_ptq = evaluate_accuracy(model, loader, device=device, max_samples=args.max_samples)
    drop = acc_baseline - acc_ptq
    validation_time = time.time() - validation_start
    print(f"\n[GATE RESULT] Baseline: {acc_baseline:.2f}% | PTQ: {acc_ptq:.2f}% | Drop: {drop:.2f}%")
    print(f"[TIMING] Validation completed in {validation_time:.2f}s ({validation_time/60:.1f} min)")

    # ---------------------------------------------------------
    # STEP 4: DECISION (Recover?)
    # ---------------------------------------------------------
    if drop <= qat_threshold:
        print(f"\n[SUCCESS] PTQ Passed! (Drop {drop:.2f}% <= {qat_threshold}%)")
        print(f"Optimal Model Ready: {joint_config_json}")

        used_qat = False
        final_acc = acc_ptq
    else:
        print(f"\n[FAILURE] PTQ Failed! (Drop {drop:.2f}% > {qat_threshold}%)")
        print("[ACTION] Triggering QAT Recovery Process...")

        # ---------------------------------------------------------
        # STEP 5: RECOVER (QAT)
        # ---------------------------------------------------------
        qat_start = time.time()
        print(f"\n[STEP 5] Running Quantization-Aware Training (QAT)...")
        cmd = f"python quantization_framework/experiments/qat_training.py " \
              f"--model {model_name} " \
              f"--checkpoint {checkpoint_path} " \
              f"--config {weight_config_json} " \
              f"--activation-config {activation_config_json} " \
              f"--dataset {dataset} " \
              f"--epochs 15 " \
              f"--patience 5 " \
              f"--max-samples {args.max_samples}"
        run_command(cmd)

        print("\n[SUCCESS] QAT Completed. Measuring final accuracy...")

        # Load QAT checkpoint
        qat_checkpoint = f"{model_name}_qat_best.pth"
        if os.path.exists(qat_checkpoint):
            print(f"[ENGINE] Loading QAT checkpoint: {qat_checkpoint}")
            model_qat = load_model(model_name, checkpoint_path=qat_checkpoint, num_classes=num_classes)
            model_qat = model_qat.to(device)

            # Apply activation quantizers only (weights already QAT-trained)
            model_qat, quantizers_qat = apply_mixed_precision(model_qat, weight_config,
                                 quantize_weights=False,
                                 quantize_activations=True,
                                 act_bit_width=8,
                                 activation_config=activation_config)

            if quantizers_qat:
                calibrate_activation_quantizers(model_qat, quantizers_qat, loader, device=device, num_batches=10)

            final_acc = evaluate_accuracy(model_qat, loader, device=device, max_samples=args.max_samples)
            print(f"[QAT RESULT] Final Accuracy: {final_acc:.2f}%")
            model = model_qat
        else:
            print(f"[WARNING] QAT checkpoint not found. Using PTQ accuracy.")
            final_acc = acc_ptq

        qat_time = time.time() - qat_start
        print(f"[TIMING] QAT completed in {qat_time:.2f}s ({qat_time/60:.1f} min)")
        used_qat = True

    # ============================================================
    # STEP 6: BOPs ANALYSIS (Joint W=A)
    # ============================================================
    print(f"\n[STEP 6] Calculating BOPs (Joint W=A)...")

    # Reconstruct joint config format from separate weight and activation configs
    joint_config_for_bops = {}
    for layer in weight_config.keys():
        joint_config_for_bops[layer] = {
            'weight': weight_config[layer],
            'activation': activation_config[layer]
        }

    # Create baseline config (all layers at 32-bit)
    baseline_config = {}
    for layer in weight_config.keys():
        baseline_config[layer] = {'weight': 32, 'activation': 32}

    # Calculate BOPs with joint W=A
    baseline_bops = calculate_bops_joint(model, baseline_config, input_size)
    quantized_bops = calculate_bops_joint(model, joint_config_for_bops, input_size)
    bops_reduction = baseline_bops / quantized_bops if quantized_bops > 0 else 1.0

    # Calculate average bits (handling granular bit-lists)
    def get_avg_val(v):
        return sum(v)/len(v) if isinstance(v, list) else v
    
    bits_list = [get_avg_val(weight_config[layer]) for layer in weight_config.keys()]
    avg_bits = sum(bits_list) / len(bits_list) if bits_list else 8.0

    print(f"\n{'='*70}")
    print(f"BOPs ANALYSIS (Joint W=A Mixed-Precision)")
    print(f"{'='*70}")
    print(f"  Baseline (FP32/FP32):     {baseline_bops:.2f} GBOPs")
    print(f"  Quantized (Mixed W=A):    {quantized_bops:.2f} GBOPs")
    print(f"  BOPs Reduction:           {bops_reduction:.2f}x")
    print(f"  Average bits (W=A):       {avg_bits:.2f}")
    print(f"{'='*70}\n")

    # Calculate total time
    total_time = time.time() - overall_start

    # Print timing summary
    print(f"{'='*70}")
    print(f"TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  Joint Profiling: {probe_time:>8.2f}s ({probe_time/60:.1f} min)")
    print(f"  Joint Search:    {search_time:>8.2f}s ({search_time/60:.1f} min)")
    print(f"  Validation:      {validation_time:>8.2f}s ({validation_time/60:.1f} min)")
    if used_qat:
        print(f"  QAT Training:    {qat_time:>8.2f}s ({qat_time/60:.1f} min)")
    else:
        print(f"  QAT Training:    Skipped (PTQ passed)")
    print(f"  {'-'*40}")
    print(f"  Total:           {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")

    # Model size info
    from models.model_loaders import get_model_size_info
    baseline_model = load_model(model_name, checkpoint_path, num_classes)
    baseline_size_mb = get_model_size_info(baseline_model)['size_mb']

    # For quantized size, estimate from bit-width
    quantized_size_mb = baseline_size_mb * (avg_bits / 32.0)

    # Save metrics
    save_metrics(model_name, acc_baseline, acc_ptq, final_acc,
                 joint_config_json, output_metrics,
                 used_qat=used_qat, qat_threshold=qat_threshold,
                 baseline_size_mb=baseline_size_mb,
                 quantized_size_mb=quantized_size_mb,
                 probe_time=probe_time, search_time=search_time,
                 validation_time=validation_time, qat_time=qat_time if used_qat else None,
                 compress_time=compress_time, total_time=total_time,
                 baseline_bops=baseline_bops, quantized_bops=quantized_bops,
                 bops_reduction=bops_reduction,
                 wa_compliance=wa_compliance, avg_bits=avg_bits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-Quantization Engine (Joint W=A)')
    parser.add_argument('--model', type=str, required=True, help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb', 'fashionmnist'],
                        help='Dataset name (default: cifar10)')
    parser.add_argument('--bits', type=int, nargs='+', default=[4, 6, 8],
                        help='List of bit-widths (default: 4 6 8, avoids 2-bit)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Target accuracy drop for search (default: 3.0%%)')
    parser.add_argument('--qat-threshold', type=float, default=5.0,
                        help='Accuracy drop threshold to trigger QAT (default: 5.0%%)')
    parser.add_argument('--output-metrics', type=str, default='metrics.json',
                        help='Output file for comprehensive metrics')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples for evaluation/profiling (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Evaluation batch size (default: 128; AlexNet auto-capped to 32)')

    # GTSRB-specific options
    parser.add_argument('--gtsrb-use-train-val', type=bool, default=False,
                        help='For GTSRB: Use internal Train folder validation split')
    parser.add_argument('--gtsrb-val-ratio', type=float, default=0.2,
                        help='For GTSRB: Validation split ratio (default: 0.2)')
    parser.add_argument('--gtsrb-seed', type=int, default=42,
                        help='For GTSRB: Random seed for train/val split (default: 42)')
    parser.add_argument('--register-size', type=int, default=16,
                        help='Hardware register size in bits (default: 16)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cpu"). Auto-detects if None.')

    args = parser.parse_args()

    # Device detection (Move to Top)
    import torch
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    auto_quantize_joint(args)
