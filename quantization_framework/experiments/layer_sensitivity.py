"""
Layer-Level Sensitivity Analysis
Measures accuracy drop when quantizing entire layers (not per-channel).
Outputs format compatible with hardware_aware_search.py
"""

import argparse
import csv
import time
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader


def run_layer_sensitivity(model_name, checkpoint_path, dataset='cifar10', 
                          output_csv='layer_profile.csv', bit_widths=[2, 4, 8]):
    """
    Measure layer-level sensitivity: quantize entire layer, measure accuracy drop.
    
    Args:
        model_name: Model architecture
        checkpoint_path: Checkpoint path
        dataset: Dataset name
        output_csv: Output CSV file
        bit_widths: Bit-widths to test
    """
    print(f"\n{'='*60}")
    print(f"LAYER-LEVEL SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Testing bit-widths: {bit_widths}")
    print(f"{'='*60}\n")

    # Initialize timing
    overall_start = time.time()
    layer_times = []

    # Load model
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10
    
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    model.to('cuda')
    model.eval()
    
    # Load data
    # Determine input size and batch size based on model and dataset
    if dataset == 'gtsrb':
        input_size = 224  # GTSRB uses larger images
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32   # CIFAR models use 32x32
    else:
        input_size = 224  # Swin/LeViT use 224x224

    # Reduce batch size for memory-intensive models (Swin, LeViT)
    if model_name == 'swin':
        batch_size = 16  # Very small batch for Swin to avoid OOM
    elif model_name == 'levit':
        batch_size = 32  # Smaller batch for LeViT
    else:
        batch_size = 128

    if dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    
    # Measure baseline
    print("Measuring baseline accuracy...")
    baseline_start = time.time()

    # Clear GPU cache before starting
    torch.cuda.empty_cache()

    try:
        baseline_acc = evaluate_accuracy(model, loader, device='cuda', max_samples=2000)
        baseline_time = time.time() - baseline_start
        print(f"Baseline: {baseline_acc:.2f}% (measured in {baseline_time:.2f}s)\n")
    except RuntimeError as e:
        if "out of memory" in str(e) or "CUBLAS" in str(e):
            print(f"GPU out of memory error. Try reducing batch size further or use CPU.")
            print(f"Error: {e}")
            raise
    
    # Get quantizable layers
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layers.append(name)
    
    print(f"Analyzing {len(layers)} layers...\n")
    
    # Store results
    results = []
    
    for idx, layer_name in enumerate(layers):
        layer_start = time.time()
        print(f"[{idx+1}/{len(layers)}] Testing {layer_name}...")

        layer_result = {
            'layer': layer_name,
            'baseline_acc': baseline_acc
        }
        
        # Test each bit-width
        for bit_width in sorted(bit_widths):
            # Create model copy
            model_copy = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
            model_copy.to('cuda')
            model_copy.eval()
            
            # Quantize ONLY this layer
            for name, module in model_copy.named_modules():
                if name == layer_name and hasattr(module, 'weight'):
                    w = module.weight.data
                    q_w, _, _ = quantize_tensor(w, bit_width=bit_width)
                    module.weight.data = q_w
            
            # Measure accuracy
            acc = evaluate_accuracy(model_copy, loader, device='cuda', max_samples=2000)
            sensitivity = baseline_acc - acc
            
            layer_result[f'accuracy_{bit_width}bit'] = round(acc, 2)
            layer_result[f'sensitivity_{bit_width}bit'] = round(sensitivity, 2)
            
            print(f"  {bit_width}-bit: Acc={acc:.2f}%, Drop={sensitivity:.2f}%")
            
            del model_copy
            torch.cuda.empty_cache()

        layer_time = time.time() - layer_start
        layer_times.append(layer_time)
        results.append(layer_result)
        print(f"  [TIMING] Layer analyzed in {layer_time:.2f}s")
        print()
    
    # Save to CSV
    if results:
        fieldnames = ['layer', 'baseline_acc']
        for bit_width in sorted(bit_widths):
            fieldnames.extend([f'accuracy_{bit_width}bit', f'sensitivity_{bit_width}bit'])
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # Calculate total time
        total_time = time.time() - overall_start
        avg_layer_time = sum(layer_times) / len(layer_times) if layer_times else 0

        print(f"{'='*60}")
        print(f"✓ Sensitivity profile saved to {output_csv}")
        print(f"{'='*60}")
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline measurement: {baseline_time:>8.2f}s")
        print(f"  Layers analyzed:      {len(layers):>8d}")
        print(f"  Avg time per layer:   {avg_layer_time:>8.2f}s")
        print(f"  Total layer analysis: {sum(layer_times):>8.2f}s")
        print(f"  {'-'*40}")
        print(f"  Total time:           {total_time:>8.2f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layer-Level Sensitivity Analysis')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (default: {model}_weight_sensitivity.csv)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8],
                        help='Bit-widths to test (e.g., --bits 2 4 8)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"{args.model}_weight_sensitivity.csv"
        print(f"No output file specified, using: {args.output}")
    
    run_layer_sensitivity(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_csv=args.output,
        bit_widths=args.bits
    )
