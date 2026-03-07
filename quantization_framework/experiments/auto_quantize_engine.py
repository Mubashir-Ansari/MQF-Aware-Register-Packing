import argparse
import os
import sys
import subprocess
import json
import time
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pipeline import get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, evaluate_accuracy
from models.model_loaders import load_model

def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"\n[ENGINE] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with return code {ret}: {cmd}")

def calculate_compression_ratio(model, config):
    """
    Calculate compression from FP32 to mixed-precision.
    Returns: compression_ratio (e.g., 4.0 = 4x compression)
    """
    total_bits_original = 0
    total_bits_quantized = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            num_elements = module.weight.numel()
            original_bits = num_elements * 32  # FP32
            
            # Get assigned bit-width from config
            if name in config:
                bits = config[name]
                
                if isinstance(bits, list):
                    # Granular: average bit-width across channels
                    avg_bits = sum(bits) / len(bits)
                    quantized_bits = num_elements * avg_bits
                else:
                    # Layer-wise: single bit-width
                    quantized_bits = num_elements * bits
            else:
                # Not quantized, remains FP32
                quantized_bits = original_bits
            
            total_bits_original += original_bits
            total_bits_quantized += quantized_bits
    
    if total_bits_quantized == 0:
        return 1.0

    return total_bits_original / total_bits_quantized


def calculate_bops(model, config, input_size=32):
    """
    Calculate Bit Operations (BOPs) for mixed-precision WEIGHT quantization.

    ASSUMPTION: Activations are FIXED at 8-bit (main pipeline design).
    For variable activation quantization, see activation_experiments/.

    BOPs = sum over layers of: MACs × weight_bits × 8

    Args:
        model: PyTorch model
        config: Mixed-precision weight config (e.g., {"layer": 4})
        input_size: Input image size for spatial dimension calculation

    Returns:
        BOPs in GigaBOPs (GBOPs)
    """
    import torch.nn as nn

    total_bops = 0
    activation_bits = 8  # FIXED - Main pipeline uses 8-bit activations

    for name, module in model.named_modules():
        if name in config:
            # Handle both int and dict formats
            if isinstance(config[name], dict):
                w_bits = config[name].get('weight', 8)
            elif isinstance(config[name], list):
                w_bits = sum(config[name]) / len(config[name])  # avg for granular
            else:
                w_bits = config[name]

            if isinstance(module, nn.Conv2d):
                stride = module.stride[0] if hasattr(module.stride, '__getitem__') else module.stride
                h_out = input_size // stride
                w_out = h_out
                macs = h_out * w_out * module.in_channels * module.out_channels * \
                       module.kernel_size[0] * module.kernel_size[1]
                total_bops += macs * w_bits * activation_bits

            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
                total_bops += macs * w_bits * activation_bits

    return total_bops / 1e9  # GBOPs


def save_metrics(model_name, baseline_acc, ptq_acc, final_acc,
                 compression_ratio, config_path, output_file,
                 used_qat=False, qat_threshold=2.0,
                 baseline_size_mb=None, quantized_size_mb=None,
                 probe_time=None, search_time=None, validation_time=None,
                 qat_time=None, compress_time=None, total_time=None,
                 baseline_bops=None, quantized_bops=None, bops_reduction=None):
    """Save comprehensive metrics to JSON"""
    from datetime import datetime

    metrics = {
        'model': model_name,
        'baseline_accuracy': round(baseline_acc, 2),
        'ptq_accuracy': round(ptq_acc, 2),
        'final_accuracy': round(final_acc, 2),
        'accuracy_drop': round(baseline_acc - final_acc, 2),
        'ptq_drop': round(baseline_acc - ptq_acc, 2),
        'compression_ratio': round(compression_ratio, 2),
        'compression_percentage': round((1 - 1/compression_ratio) * 100, 2),
        'baseline_size_mb': round(baseline_size_mb, 2) if baseline_size_mb else None,
        'quantized_size_mb': round(quantized_size_mb, 2) if quantized_size_mb else None,
        'bops_analysis': {
            'baseline_bops_gbops': round(baseline_bops, 2) if baseline_bops else None,
            'quantized_bops_gbops': round(quantized_bops, 2) if quantized_bops else None,
            'bops_reduction': round(bops_reduction, 2) if bops_reduction else None,
            'note': 'Calculated with mixed-precision weights and FIXED 8-bit activations'
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
    print(f"  Compression: {compression_ratio:.2f}x ({(1 - 1/compression_ratio) * 100:.1f}% size reduction)")
    if baseline_size_mb and quantized_size_mb:
        print(f"  Size: {baseline_size_mb:.2f} MB → {quantized_size_mb:.2f} MB")
    if baseline_bops and quantized_bops:
        print(f"  BOPs: {baseline_bops:.2f} GBOPs → {quantized_bops:.2f} GBOPs ({bops_reduction:.2f}x reduction)")


def auto_quantize(model_name, checkpoint_path, dataset='cifar10', target_drop=3.0, bit_choices=[2, 4, 8],
                  quantize_weights=True, quantize_activations=True,
                  output_metrics='metrics.json', qat_threshold=2.0,
                  gtsrb_use_train_val=False, gtsrb_val_ratio=0.2, gtsrb_seed=42):
    """
    The Unified Auto-Quantize Engine Logic.
    Flow: Probe -> Search -> Validate (Gate) -> [QAT if needed]
    
    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name (cifar10, cifar100, gtsrb)
        target_drop: Target accuracy drop for search (default: 3.0%)
        bit_choices: List of bit-widths to consider (default: [2, 4, 8])
        quantize_weights: Enable weight quantization (default: True)
        quantize_activations: Enable activation quantization (default: True)
        output_metrics: Metrics output file path (default: 'metrics.json')
        qat_threshold: Accuracy drop threshold to trigger QAT (default: 2.0%)
        gtsrb_use_train_val: Use internal Train folder split for GTSRB (default: False)
        gtsrb_val_ratio: Validation ratio for GTSRB internal split (default: 0.2)
        gtsrb_seed: Random seed for GTSRB internal split (default: 42)
    """
    print("="*60)
    print(f"AUTO-QUANTIZATION ENGINE: Processing {model_name}")
    print(f"Dataset: {dataset}")
    print(f"User Bit-Choices: {bit_choices}")
    print(f"Quantize Weights: {quantize_weights} | Quantize Activations: {quantize_activations}")
    print(f"QAT Threshold: {qat_threshold}%")
    if dataset == 'gtsrb' and gtsrb_use_train_val:
        print(f"GTSRB Mode: Internal Train/Val Split (val_ratio={gtsrb_val_ratio}, seed={gtsrb_seed})")
    print("="*60)

    # Initialize timing
    overall_start = time.time()
    probe_time = search_time = validation_time = qat_time = compress_time = 0.0

    # Paths
    profile_csv = f"{model_name}_profile.csv"
    config_json = f"{model_name}_auto_config.json"
    
    # ---------------------------------------------------------
    # STEP 1: PROBE (Sensitivity Analysis)
    # ---------------------------------------------------------
    probe_start = time.time()
    if not os.path.exists(profile_csv):
        print(f"\n[STEP 1] Generating Sensitivity Profile ({profile_csv})...")
        bits_str = " ".join(map(str, bit_choices))
        cmd = f"python quantization_framework/experiments/layer_sensitivity.py --model {model_name} --checkpoint {checkpoint_path} --output {profile_csv} --dataset {dataset} --bits {bits_str}"
        run_command(cmd)
    else:
        print(f"\n[STEP 1] Found existing profile: {profile_csv}")
    probe_time = time.time() - probe_start
    print(f"[TIMING] Profiling completed in {probe_time:.2f}s ({probe_time/60:.1f} min)")

    # ---------------------------------------------------------
    # STEP 2: SEARCH (Generate Config)
    # ---------------------------------------------------------
    search_start = time.time()
    print(f"\n[STEP 2] Searching for Optimal Configuration...")
    bits_str = " ".join(map(str, bit_choices))
    cmd = f"python quantization_framework/experiments/hardware_aware_search.py --model {model_name} --checkpoint {checkpoint_path} --profile {profile_csv} --output {config_json} --bits {bits_str} --dataset {dataset} --target-drop {target_drop}"
    run_command(cmd)
    search_time = time.time() - search_start
    print(f"[TIMING] Search completed in {search_time:.2f}s ({search_time/60:.1f} min)")
    
    # ---------------------------------------------------------
    # STEP 3: GATE (Validate PTQ)
    # ---------------------------------------------------------
    validation_start = time.time()
    print(f"\n[STEP 3] Validating PTQ Accuracy (The Gate)...")
    
    # We validate via subprocess to keep state clean
    cmd = f"python quantization_framework/experiments/validate_config.py --model {model_name} --checkpoint {checkpoint_path} --config {config_json} --dataset {dataset}"
    run_command(cmd)
    
    # Check Accuracy Internally
    print("[ENGINE] internal validation check...")
    with open(config_json, 'r') as f:
        config = json.load(f)
    
    # Determine number of classes based on dataset
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10
    
    # Load model
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    
    # Determine device (auto-detect)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ENGINE] Using device: {device}")
    
    # Move model to device BEFORE quantization
    model = model.to(device)
    
    # Determine input size based on model architecture
    if dataset == 'gtsrb':
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:  # levit, swin
        input_size = 224

    # Load appropriate dataloader
    if dataset == 'cifar100':
        loader = get_cifar100_dataloader(train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        # CRITICAL: Use correct split mode for GTSRB
        loader = get_gtsrb_dataloader(
            train=False, 
            input_size=input_size,
            use_train_val_split=gtsrb_use_train_val,
            val_ratio=gtsrb_val_ratio,
            seed=gtsrb_seed
        )
    else:  # cifar10
        loader = get_cifar10_dataloader(train=False, input_size=input_size)
    
    # Quick Baseline
    acc_baseline = evaluate_accuracy(model, loader, device=device, max_samples=1000)
    
    # Apply Config
    from experiments.validate_config import apply_mixed_precision, calibrate_activation_quantizers
    model, quantizers = apply_mixed_precision(model, config, 
                         quantize_weights=quantize_weights,
                         quantize_activations=quantize_activations,
                         act_bit_width=8)
    
    # Calibrate activation quantizers if enabled
    if quantizers:
        calibrate_activation_quantizers(model, quantizers, loader, device=device, num_batches=10)
    
    acc_ptq = evaluate_accuracy(model, loader, device=device, max_samples=1000)
    drop = acc_baseline - acc_ptq
    validation_time = time.time() - validation_start
    print(f"\n[GATE RESULT] Baseline: {acc_baseline:.2f}% | PTQ: {acc_ptq:.2f}% | Drop: {drop:.2f}%")
    print(f"[TIMING] Validation completed in {validation_time:.2f}s ({validation_time/60:.1f} min)")

    # ---------------------------------------------------------
    # STEP 4: DECISION (Recover?)
    # ---------------------------------------------------------
    if drop <= qat_threshold:
        print(f"\n[SUCCESS] PTQ Passed! (Drop {drop:.2f}% <= {qat_threshold}%)")
        print(f"Optimal Model Ready: {config_json}")
        
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
        cmd = f"python quantization_framework/experiments/qat_training.py --model {model_name} --checkpoint {checkpoint_path} --config {config_json} --dataset {dataset} --epochs 15 --patience 5"
        run_command(cmd)
        
        print("\n[SUCCESS] QAT Completed. Measuring final accuracy...")

        # Load QAT checkpoint and measure final accuracy
        qat_checkpoint = f"{model_name}_qat_best.pth"
        if os.path.exists(qat_checkpoint):
            print(f"[ENGINE] Loading QAT checkpoint: {qat_checkpoint}")
            model_qat = load_model(model_name, checkpoint_path=qat_checkpoint, num_classes=num_classes)

            # Move model to device
            model_qat = model_qat.to(device)

            # IMPORTANT: QAT checkpoint already has quantized weights from training!
            # Only apply activation quantizers, NOT weight quantization (would double-quantize)
            if quantize_activations:
                from experiments.validate_config import apply_mixed_precision, calibrate_activation_quantizers
                # Set quantize_weights=False to avoid double quantization
                model_qat, quantizers_qat = apply_mixed_precision(model_qat, config,
                                     quantize_weights=False,  # Weights already QAT-trained!
                                     quantize_activations=True,
                                     act_bit_width=8)

                # Calibrate activation quantizers
                if quantizers_qat:
                    calibrate_activation_quantizers(model_qat, quantizers_qat, loader, device=device, num_batches=10)

            # Evaluate on full validation set (not just 1000 samples)
            final_acc = evaluate_accuracy(model_qat, loader, device=device, max_samples=None)
            print(f"[QAT RESULT] Final Accuracy: {final_acc:.2f}%")
            model = model_qat
        else:
            print(f"[WARNING] QAT checkpoint not found at {qat_checkpoint}. Using PTQ accuracy as fallback.")
            final_acc = acc_ptq

        qat_time = time.time() - qat_start
        print(f"[TIMING] QAT completed in {qat_time:.2f}s ({qat_time/60:.1f} min)")
        used_qat = True

    # ============================================================
    # STEP 6: ACTUAL COMPRESSION
    # ============================================================
    compress_start = time.time()
    print(f"\n[STEP 6] Compressing Model to Disk...")

    from export.compress_model import compress_model

    compressed_model_path = f"{model_name}_compressed.pkl"
    actual_compressed_size_mb = compress_model(model, config, compressed_model_path)
    compress_time = time.time() - compress_start
    print(f"[TIMING] Compression completed in {compress_time:.2f}s")
    
    # Update metrics with ACTUAL sizes
    from models.model_loaders import get_model_size_info
    baseline_model = load_model(model_name, checkpoint_path, num_classes)
    baseline_size_mb = get_model_size_info(baseline_model)['size_mb']
    actual_compression_ratio = baseline_size_mb / actual_compressed_size_mb
    
    print(f"\n{'='*60}")
    print(f"FINAL COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline Size:      {baseline_size_mb:.2f} MB")
    print(f"  Compressed Size:    {actual_compressed_size_mb:.2f} MB")
    print(f"  Actual Compression: {actual_compression_ratio:.2f}x")
    print(f"  Size Reduction:     {(1 - actual_compressed_size_mb/baseline_size_mb)*100:.1f}%")
    print(f"{'='*60}\n")

    # ============================================================
    # BOPs Analysis (Weight Quantization + Fixed 8-bit Activations)
    # ============================================================
    print(f"[ENGINE] Calculating BOPs...")

    # Create baseline config (all layers at 32-bit)
    baseline_config = {layer: 32 for layer in config.keys()}

    # Calculate BOPs
    baseline_bops = calculate_bops(baseline_model, baseline_config, input_size)
    quantized_bops = calculate_bops(baseline_model, config, input_size)
    bops_reduction = baseline_bops / quantized_bops if quantized_bops > 0 else 1.0

    print(f"\n{'='*60}")
    print(f"BOPs ANALYSIS (Mixed-Precision Weights + 8-bit Activations)")
    print(f"{'='*60}")
    print(f"  Baseline (FP32/FP32):     {baseline_bops:.2f} GBOPs")
    print(f"  Quantized (Mixed-W/A8):   {quantized_bops:.2f} GBOPs")
    print(f"  BOPs Reduction:           {bops_reduction:.2f}x")
    print(f"{'='*60}\n")

    # Calculate total time
    total_time = time.time() - overall_start

    # Print timing summary
    print(f"{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Profiling:    {probe_time:>8.2f}s ({probe_time/60:.1f} min)")
    print(f"  Search:       {search_time:>8.2f}s ({search_time/60:.1f} min)")
    print(f"  Validation:   {validation_time:>8.2f}s ({validation_time/60:.1f} min)")
    if used_qat:
        print(f"  QAT Training: {qat_time:>8.2f}s ({qat_time/60:.1f} min)")
    else:
        print(f"  QAT Training:  Skipped (PTQ passed)")
    print(f"  Compression:  {compress_time:>8.2f}s")
    print(f"  {'-'*40}")
    print(f"  Total:        {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    # Save updated metrics
    save_metrics(model_name, acc_baseline, acc_ptq, final_acc,
                 actual_compression_ratio, config_json, output_metrics,
                 used_qat=used_qat, qat_threshold=qat_threshold,
                 baseline_size_mb=baseline_size_mb,
                 quantized_size_mb=actual_compressed_size_mb,
                 probe_time=probe_time, search_time=search_time,
                 validation_time=validation_time, qat_time=qat_time if used_qat else None,
                 compress_time=compress_time, total_time=total_time,
                 baseline_bops=baseline_bops, quantized_bops=quantized_bops,
                 bops_reduction=bops_reduction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Auto-Quantization Engine')
    parser.add_argument('--model', type=str, required=True, help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (cifar10/cifar100/gtsrb)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8], 
                        help='List of bit-widths (e.g. --bits 2 4 8)')
    
    # Boolean parsing helper
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('--quantize-weights', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Enable weight quantization (default: True). Usage: --quantize-weights false')
    parser.add_argument('--quantize-activations', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Enable activation quantization (default: True). Usage: --quantize-activations false')
    parser.add_argument('--output-metrics', type=str, default='metrics.json',
                        help='Output file for comprehensive metrics')
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Target accuracy drop for search (default: 3.0%%)')
    parser.add_argument('--qat-threshold', type=float, default=5.0,
                        help='Accuracy drop threshold to trigger QAT (default: 5.0%%)')
    
    # GTSRB-specific options (NEW)
    parser.add_argument('--gtsrb-use-train-val', type=str_to_bool, nargs='?', const=True, default=False,
                        help='For GTSRB: Use internal Train folder validation split instead of Test.csv. '
                             'Set to TRUE if your model was trained on Train folder split. (default: False)')
    parser.add_argument('--gtsrb-val-ratio', type=float, default=0.2,
                        help='For GTSRB: Validation split ratio when using train/val split (default: 0.2)')
    parser.add_argument('--gtsrb-seed', type=int, default=42,
                        help='For GTSRB: Random seed for train/val split - MUST match training! (default: 42)')
    
    args = parser.parse_args()
    
    auto_quantize(args.model, args.checkpoint, args.dataset, 
                  target_drop=args.target_drop,
                  bit_choices=args.bits,
                  quantize_weights=args.quantize_weights,
                  quantize_activations=args.quantize_activations,
                  output_metrics=args.output_metrics,
                  qat_threshold=args.qat_threshold,
                  gtsrb_use_train_val=args.gtsrb_use_train_val,
                  gtsrb_val_ratio=args.gtsrb_val_ratio,
                  gtsrb_seed=args.gtsrb_seed)
