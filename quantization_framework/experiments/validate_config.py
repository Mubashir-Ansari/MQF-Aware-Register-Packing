import argparse
import json
import time
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from quantization.activations import ActivationQuantizer
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, get_fashionmnist_dataloader
import torch.nn as nn
import sys
import models.alexnet
# Fix for checkpoint loading (fasion_mnist_alexnet class mismatch)
fasion_mnist_alexnet = models.alexnet.AlexNet
sys.modules['__main__'].fasion_mnist_alexnet = models.alexnet.AlexNet


def insert_activation_quantizers(model, act_bit_width=8, quantize_activations=True,
                                  activation_config=None):
    """
    Insert ActivationQuantizer modules after key layers.
    Modifies model in-place by wrapping activations using forward hooks.
    
    Args:
        model: PyTorch model to modify
        act_bit_width: Default bit-width for activation quantization (default: 8)
        quantize_activations: Whether to enable activation quantization (default: True)
        activation_config: Optional dict {layer_name: activation_bits} for per-layer (NEW!)
    
    Returns:
        model: Modified model with activation quantizers attached
        quantizers: List of quantizer instances for calibration
    """
    if not quantize_activations:
        print("[INFO] Activation quantization disabled.")
        return model, []
    
    # IMPORTANT: Clean up existing hooks to prevent stacking
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
    
    # Track which layers to quantize
    layers_to_wrap = []
    
    for name, module in model.named_modules():
        # Quantize activations after Conv, Linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers_to_wrap.append((name, module))
    
    # Insert quantizers using forward hooks
    quantizers = []
    # Detect model's device to ensure quantizers match
    model_device = next(model.parameters()).device
    
    # Track per-layer bit-widths for logging
    layer_bit_dist = {}
    channel_bit_dist = {}
    
    for name, module in layers_to_wrap:
        # Determine bit-width for this layer
        if activation_config and name in activation_config:
            bits = activation_config[name]  # Per-layer or List
        else:
            bits = act_bit_width  # Default
        
        # Get number of output channels for granular quantization
        num_channels = None
        if hasattr(module, 'weight') and module.weight is not None:
            num_channels = module.weight.shape[0]
        elif hasattr(module, 'out_features'):
            num_channels = module.out_features
            
        quantizer = ActivationQuantizer(bit_width=bits, num_channels=num_channels)
        quantizer.to(model_device)  # Move to model's device
        quantizers.append(quantizer)
        
        # Track distribution
        if isinstance(bits, list):
            # Layer-level dominant representation for readable percentages
            from collections import Counter
            counts = Counter(bits)
            dominant_bit = counts.most_common(1)[0][0] if counts else act_bit_width
            layer_bit_dist[dominant_bit] = layer_bit_dist.get(dominant_bit, 0) + 1

            # Channel-level view (optional, does not sum to number of layers)
            for b in bits:
                channel_bit_dist[b] = channel_bit_dist.get(b, 0) + 1
        else:
            layer_bit_dist[bits] = layer_bit_dist.get(bits, 0) + 1
        
        # Create closure to capture quantizer instance
        def make_hook(q):
            def hook(module, input, output):
                return q(output)
            return hook
        
        module.register_forward_hook(make_hook(quantizer))
    
    # Log summary
    if activation_config:
        print(f"[ACTIVATION QUANT] Inserted {len(quantizers)} mixed-precision quantizers:")
        for bits in sorted(layer_bit_dist.keys(), reverse=True):
            count = layer_bit_dist[bits]
            pct = (count / len(quantizers)) * 100
            print(f"  A{bits}: {count:3d} layers ({pct:5.1f}%)")
        if channel_bit_dist:
            total_channels = sum(channel_bit_dist.values())
            print("  Channel-level distribution:")
            for bits in sorted(channel_bit_dist.keys(), reverse=True):
                count = channel_bit_dist[bits]
                pct = (count / total_channels) * 100 if total_channels > 0 else 0.0
                print(f"    A{bits}: {count:4d} channels ({pct:5.1f}%)")
    else:
        print(f"[ACTIVATION QUANT] Inserted quantizers on {len(quantizers)} layers (bit_width={act_bit_width})")
    
    return model, quantizers


def calibrate_activation_quantizers(model, quantizers, dataloader, device=None, num_batches=10):
    """
    Calibrate activation quantizers by running inference on calibration data.
    Device-agnostic: Auto-detects model's device if not specified.
    
    Args:
        model: Model with quantizers attached
        quantizers: List of ActivationQuantizer instances
        dataloader: Calibration data loader
        device: Device to run on (auto-detected if None)
        num_batches: Number of batches to use for calibration (default: 10)
    """
    if not quantizers:
        return
    
    # Auto-detect device if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Auto-detect device
            if not torch.cuda.is_available():
                device = torch.device('cpu')
            else:
                device = torch.device('cuda') # Default to cuda if available and no model params yet
    
    print(f"Using device: {device}")
    
    print(f"[CALIBRATION] Calibrating activation quantizers with {num_batches} batches on {device}...")
    
    # Put quantizers in training mode to collect statistics
    for q in quantizers:
        q.train()
        q.to(device)  # Ensure quantizers are on correct device
    
    # Put model in eval mode
    model.eval()
    
    # Run calibration
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    
    # Put quantizers in eval mode (use collected statistics)
    for q in quantizers:
        q.eval()
    
    print(f"[CALIBRATION] Complete. Quantizers calibrated.")


def apply_mixed_precision(model, config, device='cuda', 
                          quantize_weights=True, quantize_activations=True,
                          act_bit_width=8, activation_config=None):
    """
    Apply the bit-width configuration to model weights and activations.
    Supports both:
    - Layer-wise: config[layer] = int (e.g., 4)
    - Granular: config[layer] = list of ints (e.g., [2, 4, 2, 8, ...])
    
    Args:
        model: PyTorch model
        config: Bit-width configuration dict
        device: Device to place quantized weights (default: 'cuda')
        quantize_weights: Enable weight quantization (default: True)
        quantize_activations: Enable activation quantization (default: True)
        act_bit_width: Default bit-width for activations (default: 8)
        activation_config: Optional per-layer activation bit-widths dict (NEW!)
    
    Returns:
        model: Modified model
        quantizers: List of activation quantizers (empty if disabled)
    """
    if quantize_weights:
        print("Applying mixed-precision configuration to weights...")
        count = 0
        granular_count = 0
        
        for name, module in model.named_modules():
            if name in config:
                bits = config[name]
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    
                    if isinstance(bits, list):
                        # GRANULAR MODE: Per-channel quantization
                        # bits is a list like [2, 4, 2, 8, ...]
                        # Use quantize_tensor directly with list and channel_dim=0
                        q_w, _, _ = quantize_tensor(w, bit_width=bits, channel_dim=0)
                        
                        module.weight.data = q_w.to(device)
                        granular_count += 1
                    else:
                        # LAYER-WISE MODE: Single bit-width for entire layer
                        q_w, scale, zero = quantize_tensor(w, bit_width=bits)
                        module.weight.data = q_w.to(device)
                    
                    count += 1
                    
        print(f"Applied quantization to {count} layers ({granular_count} granular).")
    else:
        print("[INFO] Weight quantization disabled.")
    
    # Apply activation quantization
    quantizers = []
    if quantize_activations:
        model, quantizers = insert_activation_quantizers(model, act_bit_width, 
                                                          quantize_activations, 
                                                          activation_config)
    
    return model, quantizers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, required=True, help='Path to weight bit-width config json')
    parser.add_argument('--activation-config', type=str, default=None,
                        help='Optional per-layer activation config json (NEW!)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb', 'fashionmnist'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--input-size', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples for evaluation (default: None, full set)')
    
    args = parser.parse_args()

    # 1. Device Handling
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Initialize timing
    overall_start = time.time()

    # Auto-adjust batch size for memory-intensive models
    if args.model == 'swin' and args.batch_size > 16:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 16 for Swin Transformer (GPU memory)")
        args.batch_size = 16
    elif args.model == 'levit' and args.batch_size > 32:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 32 for LeViT (GPU memory)")
        args.batch_size = 32

    # 1. Load Configs
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load optional activation config
    activation_config = None
    if args.activation_config:
        print(f"Loading activation config: {args.activation_config}")
        with open(args.activation_config, 'r') as f:
            activation_config = json.load(f)
        print(f"  Loaded {len(activation_config)} activation bit-widths\n")
    
    # VALIDATE CONFIG FORMAT
    print("Validating config format...")
    granular_count = 0
    layer_wise_count = 0
    
    for layer_name, bits in config.items():
        if isinstance(bits, list):
            granular_count += 1
            print(f"  WARNING: Layer '{layer_name}' has granular config (list of {len(bits)} values)")
        elif isinstance(bits, int):
            layer_wise_count += 1
        else:
            raise ValueError(f"Invalid config for layer '{layer_name}': {bits}")
    
    print(f"Config summary: {layer_wise_count} layer-wise, {granular_count} granular")
    
    if granular_count > 0:
        print("\n⚠️  WARNING: Config contains granular (per-channel) quantization!")
        print("This may cause severe accuracy degradation.")
        print("Recommended: Use hardware_aware_search.py for layer-wise configs.\n")
        
    # 2. Load Model
    print(f"Loading {args.model}...")
    load_start = time.time()
    if args.dataset == 'cifar100': num_classes = 100
    elif args.dataset == 'gtsrb': num_classes = 43
    elif args.dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10

    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model.to(args.device)
    load_time = time.time() - load_start
    print(f"[TIMING] Model loaded in {load_time:.2f}s")

    # 3. Apply Config
    quant_start = time.time()
    model, quantizers = apply_mixed_precision(model, config, device=args.device,
                                   quantize_weights=True,
                                   quantize_activations=True,
                                   act_bit_width=8,
                                   activation_config=activation_config)
    quant_time = time.time() - quant_start
    print(f"[TIMING] Quantization applied in {quant_time:.2f}s")

    # 4. Load Data
    if args.input_size:
        input_size = args.input_size
    elif args.model == 'alexnet':
        input_size = 227
    elif args.dataset == 'gtsrb':
        input_size = 224  # GTSRB uses 224x224 for all models
    elif args.model in ['levit', 'swin']:
        input_size = 224
    elif args.dataset == 'fashionmnist':
        input_size = 28
    else:
        input_size = 32  # CIFAR-10/100 default
    
    print(f"Loading {args.dataset} (Input: {input_size}x{input_size})...")
    if args.dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'fashionmnist':
        loader = get_fashionmnist_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    
    # 5. Calibrate activation quantizers if enabled
    calib_time = 0
    if quantizers:
        calib_start = time.time()
        calibrate_activation_quantizers(model, quantizers, loader, device=args.device, num_batches=10)
        calib_time = time.time() - calib_start
        print(f"[TIMING] Calibration completed in {calib_time:.2f}s")

    # 6. Evaluate
    print("Measuring accuracy of Mixed-Precision Model...")
    eval_start = time.time()
    acc = evaluate_accuracy(model, loader, device=args.device, max_samples=args.max_samples)
    eval_time = time.time() - eval_start
    print(f"Final Mixed-Precision Accuracy: {acc:.2f}%")

    # Timing summary
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Model loading:    {load_time:>8.2f}s")
    print(f"  Quantization:     {quant_time:>8.2f}s")
    print(f"  Calibration:      {calib_time:>8.2f}s")
    print(f"  Evaluation:       {eval_time:>8.2f}s")
    print(f"  {'-'*40}")
    print(f"  Total:            {total_time:>8.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")
