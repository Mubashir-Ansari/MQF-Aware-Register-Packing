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
from analysis.register_packing_optimizer import run_packing_analysis


def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"\n[ENGINE] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with return code {ret}: {cmd}")


def profile_needs_regeneration(profile_csv, current_baseline_acc, required_samples=None, baseline_tol=5.0):
    """
    Detect stale/corrupt sensitivity profile.

    Regenerate if:
    1) File missing/empty
    2) Baseline in CSV differs too much from current measured baseline
    3) All/near-all sensitivity entries are zero (common bad-profile symptom)
    """
    if not os.path.exists(profile_csv):
        return True, "profile_missing"

    if required_samples is not None:
        meta_path = f"{profile_csv}.meta.json"
        if not os.path.exists(meta_path):
            return True, "profile_meta_missing"
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            prof_samples = int(meta.get("max_samples", 0))
            if prof_samples < int(required_samples):
                return True, f"profile_samples_too_low({prof_samples}<{required_samples})"
        except Exception:
            return True, "profile_meta_invalid"

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


def write_profile_metadata(profile_csv, max_samples, baseline_acc, bits):
    meta_path = f"{profile_csv}.meta.json"
    payload = {
        "max_samples": int(max_samples),
        "baseline_acc": float(baseline_acc),
        "bits": list(sorted(set(int(b) for b in bits))),
        "generated_by": "auto_quantize_engine_joint"
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)


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
    else:
        loader = get_cifar10_dataloader(batch_size=eval_batch_size, train=False, input_size=input_size)
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
    profile_samples = max(int(getattr(args, 'profile_samples', 10000)), int(args.max_samples))
    regen, reason = profile_needs_regeneration(
        joint_profile_csv,
        acc_baseline,
        required_samples=profile_samples
    )
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
              f"--max-samples {profile_samples}"
        # Note: joint_sensitivity.py auto-generates output filename
        run_command(cmd)
        write_profile_metadata(joint_profile_csv, profile_samples, acc_baseline, bit_choices)
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
          f"--max-layer-budget-share {getattr(args, 'max_layer_budget_share', 0.35)} " \
          f"--min-layers-to-modify {getattr(args, 'min_layers_to_modify', 3)} " \
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
    # STEP 3.1: HARDWARE REPORTING NOTE
    # ---------------------------------------------------------
    print("\n[STEP 3.1] Hardware Reporting")
    print("\n[BIT DISTRIBUTION SUMMARY]")
    report_bits = sorted(set(int(b) for b in bit_choices))
    bit_headers = " | ".join([f"{b}b %".rjust(7) for b in report_bits])
    print(f"{'Layer':12} | {bit_headers} | {'Status'}")
    print("-" * (28 + 11 * len(report_bits)))

    for layer, c in joint_config_data['config'].items():
        dist = c.get('granular_dist', {})
        bit_percentages = {int(b): 0.0 for b in report_bits}
        w_bit = c.get('weight', 8)

        if not dist:
            if isinstance(w_bit, list):
                total = len(w_bit)
                if total > 0:
                    for b in report_bits:
                        bit_percentages[b] = 100.0 * sum(1 for x in w_bit if int(x) == b) / total
            else:
                bw = int(w_bit) if isinstance(w_bit, int) else 8
                if bw in bit_percentages:
                    bit_percentages[bw] = 100.0
        else:
            for bits_s, pct in dist.items():
                try:
                    bits_i = int(bits_s)
                except ValueError:
                    continue
                if bits_i in bit_percentages:
                    bit_percentages[bits_i] = float(pct)

        status = "GRANULAR" if isinstance(w_bit, list) or len(dist) > 1 else "UNIFORM"
        bit_cols = " | ".join([f"{bit_percentages[b]:>6.1f}%" for b in report_bits])
        print(f"{layer:12} | {bit_cols} | {status}")

    print("-" * (28 + 11 * len(report_bits)))
    if getattr(args, "run_packing_analysis", False):
        print("  Legacy HRP-only register reporting is disabled.")
        print("  Primary hardware results will be reported in Step 7 from the")
        print("  post-MQF packing analysis module using storage + packed-issue metrics.")
    else:
        print("  Legacy HRP-only register reporting is disabled.")
        print("  Use --run-packing-analysis to generate the primary hardware report")
        print("  with weight, activation, output, and packed-issue summaries.")

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

    # ============================================================
    # OPTIONAL STEP 7: POST-MQF REGISTER PACKING ANALYSIS
    # ============================================================
    if getattr(args, "run_packing_analysis", False):
        print(f"\n[STEP 7] Running post-MQF register packing analysis...")
        packing_output_dir = getattr(args, "packing_output_dir", "packing_reports")
        aligned_policy = {}
        aligned_policy_str = getattr(args, "aligned_policy", "2:2,3:4,4:4,8:8")
        for pair in aligned_policy_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split(":")
            aligned_policy[int(k)] = int(v)

        try:
            reports = run_packing_analysis(
                model=model,
                model_name=model_name,
                dataset=dataset,
                weight_config=weight_config,
                activation_config=activation_config,
                output_dir=packing_output_dir,
                register_size=getattr(args, "register_size", 16),
                acc_width=getattr(args, "acc_width", 32),
                aligned_policy=aligned_policy,
                alpha=getattr(args, "cost_alpha", 1.0),
                beta=getattr(args, "cost_beta", 1.0),
                gamma=getattr(args, "cost_gamma", 1.0),
                delta=getattr(args, "cost_delta", 1.0),
                default_input_bits=getattr(args, "default_input_bits", 8),
                device=device,
            )
            print("[PACKING] Completed. Storage + compute summary:")
            for sname in ["raw_homogeneous", "aligned", "heterogeneous_storage", "hybrid_storage_compute"]:
                r = reports[sname]
                print(
                    f"  {sname:24s} | operand_words={r.total_registers:,} | "
                    f"storage_savings={r.savings_percent:.2f}% | packed_issue_reduction={r.total_reduction_factor:.3f}x"
                )
            base_r = reports["baseline_uniform_8bit"]
            best_name = min(
                ["raw_homogeneous", "aligned", "heterogeneous_storage", "hybrid_storage_compute"],
                key=lambda n: reports[n].objective_cost
            )
            best_r = reports[best_name]
            print(f"  baseline_uniform_8bit      | operand_words={base_r.total_registers:,} | storage_savings=0.00% | packed_issue_reduction={base_r.total_reduction_factor:.3f}x")
            print(f"[PACKING] Best strategy by objective: {best_name}")
            print(f"[PACKING] Best strategy totals: operand_words={best_r.total_registers:,}, packed_issue_reduction={best_r.total_reduction_factor:.3f}x")
            print(f"[PACKING] Reports saved to: {packing_output_dir}")
        except Exception as e:
            print(f"[PACKING] WARNING: analysis failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-Quantization Engine (Joint W=A)')
    parser.add_argument('--model', type=str, required=True, help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb', 'fashionmnist'],
                        help='Dataset name (default: cifar10)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3, 4, 8],
                        help='List of bit-widths (default: 2 3 4 8)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Target accuracy drop for search (default: 3.0%%)')
    parser.add_argument('--qat-threshold', type=float, default=5.0,
                        help='Accuracy drop threshold to trigger QAT (default: 5.0%%)')
    parser.add_argument('--output-metrics', type=str, default='metrics.json',
                        help='Output file for comprehensive metrics')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples for evaluation/profiling (default: 1000)')
    parser.add_argument('--profile-samples', type=int, default=10000,
                        help='Min samples for sensitivity profile generation (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Evaluation batch size (default: 128; AlexNet auto-capped to 32)')
    parser.add_argument('--max-layer-budget-share', type=float, default=0.35,
                        help='Max fraction of target-drop spent in one layer during search (default: 0.35)')
    parser.add_argument('--min-layers-to-modify', type=int, default=3,
                        help='Minimum desired number of modified layers in search (default: 3)')

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
    parser.add_argument('--run-packing-analysis', action='store_true',
                        help='Run post-MQF register packing strategy analysis.')
    parser.add_argument('--packing-output-dir', type=str, default='packing_reports',
                        help='Output directory for packing analysis reports.')
    parser.add_argument('--acc-width', type=int, default=32,
                        help='Accumulator width for packing analysis (default: 32).')
    parser.add_argument('--aligned-policy', type=str, default='2:2,3:4,4:4,8:8',
                        help='Aligned slot policy map, e.g. 2:2,3:4,4:4,8:8.')
    parser.add_argument('--cost-alpha', type=float, default=1.0,
                        help='Cost weight alpha for weight registers.')
    parser.add_argument('--cost-beta', type=float, default=1.0,
                        help='Cost weight beta for activation registers.')
    parser.add_argument('--cost-gamma', type=float, default=1.0,
                        help='Cost weight gamma for output registers.')
    parser.add_argument('--cost-delta', type=float, default=1.0,
                        help='Cost weight delta for packed issue count.')
    parser.add_argument('--default-input-bits', type=int, default=8,
                        help='Default input activation bits for first layer in packing analysis.')

    args = parser.parse_args()

    # Device detection (Move to Top)
    import torch
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    auto_quantize_joint(args)
