"""
Comprehensive Ablation Study: Systematic W x A Bit-Width Exploration

Tests all combinations of:
- Weight quantization: 2, 4, 6, 8 bits (uniform)
- Activation quantization: 2, 4, 6, 8 bits (uniform)
- Mixed-precision: W-Mixed, A-Mixed, and fully mixed

Total: 25 experiments (16 uniform + 4 mixed-W + 3 mixed-A + 1 fully-mixed + FP32 baseline)

Usage:
    python run_ablation_study.py \
        --model vgg11_bn \
        --dataset cifar10 \
        --output-dir ablation_results/
"""

import argparse
import json
import time
import os
import sys

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)
from quantization.primitives import quantize_tensor


class CalibrationActivationQuantizer(nn.Module):
    """Activation quantizer with calibration support."""
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.calibrating = False
        self.calibrated = False
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def calibrate_mode(self, enable=True):
        self.calibrating = enable
        if not enable and self.min_val.item() < float('inf'):
            self._compute_params()
            self.calibrated = True

    def _compute_params(self):
        q_min, q_max = 0, 2 ** self.bit_width - 1
        min_val, max_val = self.min_val.item(), self.max_val.item()
        scale = max((max_val - min_val) / (q_max - q_min), 1e-8)
        zero_point = max(q_min, min(q_max, round(-min_val / scale)))
        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)

    def forward(self, x):
        if self.scale.device != x.device:
            self.min_val = self.min_val.to(x.device)
            self.max_val = self.max_val.to(x.device)
            self.scale = self.scale.to(x.device)
            self.zero_point = self.zero_point.to(x.device)

        if self.calibrating:
            with torch.no_grad():
                self.min_val = torch.min(self.min_val, x.min())
                self.max_val = torch.max(self.max_val, x.max())
            return x

        if not self.calibrated:
            return x

        q_min, q_max = 0, 2 ** self.bit_width - 1
        x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
        return (x_int - self.zero_point) * self.scale


class MixedPrecisionValidator:
    """Applies mixed-precision quantization for both weights and activations."""

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.activation_quantizers = {}
        self.hooks = []

    def apply_weight_quantization(self):
        """Apply weight quantization according to config."""
        for name, module in self.model.named_modules():
            if name in self.config and hasattr(module, 'weight') and module.weight is not None:
                cfg = self.config[name]
                bits = cfg['weight'] if isinstance(cfg, dict) else cfg
                if bits < 32:  # Skip FP32
                    w = module.weight.data
                    q_w, _, _ = quantize_tensor(w, bit_width=bits)
                    module.weight.data = q_w

    def setup_activation_quantizers(self):
        """Install activation quantizers as forward hooks."""
        for name, module in self.model.named_modules():
            if name in self.config:
                cfg = self.config[name]
                if isinstance(cfg, dict) and 'activation' in cfg:
                    bits = cfg['activation']
                    if bits < 32:  # Skip FP32
                        quantizer = CalibrationActivationQuantizer(bit_width=bits)
                        self.activation_quantizers[name] = quantizer

                        def make_hook(quant):
                            def hook(mod, inp, out):
                                return quant(out)
                            return hook

                        handle = module.register_forward_hook(make_hook(quantizer))
                        self.hooks.append(handle)

    def calibrate(self, dataloader, num_batches=50):
        """Calibrate activation quantizers."""
        for q in self.activation_quantizers.values():
            q.calibrate_mode(True)

        self.model.eval()
        with torch.no_grad():
            for idx, (images, _) in enumerate(dataloader):
                if idx >= num_batches:
                    break
                images = images.to(self.device)
                _ = self.model(images)

        for q in self.activation_quantizers.values():
            q.calibrate_mode(False)

    def remove_hooks(self):
        """Remove all activation hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.activation_quantizers = {}


def calculate_bops(model, config, input_size=32):
    """
    Calculate Bit Operations (BOPs) for a mixed-precision configuration.

    BOPs = sum over layers of: MACs * weight_bits * activation_bits

    Returns:
        BOPs in GigaBOPs (GBOPs)
    """
    total_bops = 0

    for name, module in model.named_modules():
        if name in config:
            cfg = config[name]
            w_bits = cfg['weight'] if isinstance(cfg, dict) else cfg
            a_bits = cfg.get('activation', 8) if isinstance(cfg, dict) else 8

            if isinstance(module, nn.Conv2d):
                h_out = input_size // (module.stride[0] if hasattr(module.stride, '__getitem__') else module.stride)
                w_out = h_out
                macs = h_out * w_out * module.in_channels * module.out_channels * \
                       module.kernel_size[0] * module.kernel_size[1]
                total_bops += macs * w_bits * a_bits

            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
                total_bops += macs * w_bits * a_bits

    return total_bops / 1e9  # Convert to GBOPs


# ============================================================================
# Architecture Detection and Layer Analysis
# ============================================================================

def detect_architecture(model, model_name):
    """
    Detect architecture type from model name and structure.

    Args:
        model: PyTorch model
        model_name: Model name string

    Returns:
        Architecture type: 'vgg', 'resnet', 'levit', 'swin', or 'generic'
    """
    model_name_lower = model_name.lower()

    # Check by name first
    if 'levit' in model_name_lower:
        return 'levit'
    elif 'swin' in model_name_lower:
        return 'swin'
    elif 'vgg' in model_name_lower:
        return 'vgg'
    elif 'resnet' in model_name_lower or 'resnext' in model_name_lower:
        return 'resnet'

    # Check by structure (fallback)
    has_attention = False
    has_residual = False

    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'attn' in name_lower or 'attention' in name_lower:
            has_attention = True
        if 'downsample' in name_lower or 'residual' in name_lower:
            has_residual = True

    if has_attention:
        return 'transformer'
    elif has_residual:
        return 'resnet'
    else:
        return 'vgg'  # Default to sequential CNN


def analyze_layer_properties(module, name):
    """
    Analyze layer properties for sensitivity estimation.

    Args:
        module: PyTorch module
        name: Layer name string

    Returns:
        dict with layer_type, sparsity, param_count, is_sensitive
    """
    properties = {
        'layer_type': 'unknown',
        'sparsity': 0.0,
        'param_count': 0,
        'is_sensitive': False
    }

    if not hasattr(module, 'weight') or module.weight is None:
        return properties

    # Calculate sparsity
    properties['sparsity'] = (module.weight.data == 0).float().mean().item()
    properties['param_count'] = module.weight.numel()

    # Classify layer type
    name_lower = name.lower()

    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels and module.groups > 1:
            properties['layer_type'] = 'depthwise_conv'
            properties['is_sensitive'] = True
        elif module.kernel_size == (1, 1):
            properties['layer_type'] = 'pointwise_conv'
        else:
            properties['layer_type'] = 'standard_conv'

    elif isinstance(module, nn.Linear):
        if any(x in name_lower for x in ['qkv', 'query', 'key', 'value', 'attn']):
            properties['layer_type'] = 'attention_qkv'
            properties['is_sensitive'] = True
        elif 'proj' in name_lower:
            properties['layer_type'] = 'projection'
            properties['is_sensitive'] = True
        elif 'head' in name_lower or 'classifier' in name_lower:
            properties['layer_type'] = 'classifier'
            properties['is_sensitive'] = True
        else:
            properties['layer_type'] = 'mlp_linear'

    # Sparse layers are sensitive
    if properties['sparsity'] > 0.5:
        properties['is_sensitive'] = True

    return properties


# ============================================================================
# Architecture-Specific Bit Assignment Functions
# ============================================================================

def assign_bits_vgg(layer_idx, total_layers, name, props, weight_strategy, activation_bits):
    """
    VGG-specific bit assignment.

    VGG characteristics:
    - Sequential CNN: conv -> pool -> conv -> pool -> ... -> fc
    - Early conv layers extract basic features
    - FC layers are relatively robust
    """
    if weight_strategy == 'mixed':
        # Sparse protection
        if props['sparsity'] > 0.5:
            w_bits = 8
        # Protect first 2 and last 2 layers
        elif layer_idx < 2 or layer_idx >= total_layers - 2:
            w_bits = 8
        # Early feature extraction (first ~30% of layers)
        elif layer_idx < total_layers * 0.3:
            w_bits = 6
        # Middle conv layers
        elif props['layer_type'] == 'standard_conv':
            w_bits = 4
        # FC layers (more robust)
        elif props['layer_type'] in ['mlp_linear', 'classifier']:
            if layer_idx >= total_layers - 3:
                w_bits = 6
            else:
                w_bits = 4
        else:
            w_bits = 4
    else:
        w_bits = weight_strategy

    # Activation bits
    if activation_bits == 'mixed':
        if layer_idx < 2 or layer_idx >= total_layers - 2:
            a_bits = 8
        elif layer_idx < 5:
            a_bits = 6
        else:
            a_bits = 6
    else:
        a_bits = activation_bits

    return w_bits, a_bits


def assign_bits_resnet(layer_idx, total_layers, name, props, weight_strategy, activation_bits):
    """
    ResNet-specific bit assignment.

    ResNet characteristics:
    - Residual blocks with skip connections
    - Downsample layers (1x1 conv in skip path) are CRITICAL
    - Must maintain precision in skip connections
    """
    name_lower = name.lower()

    if weight_strategy == 'mixed':
        # Sparse protection
        if props['sparsity'] > 0.5:
            w_bits = 8
        # Initial conv (layer 0) and final FC (last layer)
        elif layer_idx == 0 or layer_idx == total_layers - 1:
            w_bits = 8
        # Downsample layers in skip connections (CRITICAL!)
        elif 'downsample' in name_lower:
            w_bits = 8
        # First stage (layer1)
        elif 'layer1' in name_lower:
            w_bits = 6
        # Middle/later stages
        elif 'layer2' in name_lower:
            w_bits = 6
        elif 'layer3' in name_lower or 'layer4' in name_lower:
            w_bits = 4
        else:
            w_bits = 6
    else:
        w_bits = weight_strategy

    # Activation bits
    if activation_bits == 'mixed':
        if layer_idx == 0 or layer_idx == total_layers - 1 or 'downsample' in name_lower:
            a_bits = 8
        elif 'layer1' in name_lower:
            a_bits = 8
        elif 'layer2' in name_lower:
            a_bits = 6
        else:
            a_bits = 6
    else:
        a_bits = activation_bits

    return w_bits, a_bits


def assign_bits_levit(layer_idx, total_layers, name, props, weight_strategy, activation_bits):
    """
    LeViT-specific bit assignment.

    LeViT characteristics:
    - Hybrid CNN + Transformer
    - Attention layers (Q/K/V) are SENSITIVE
    - MLP can be more aggressive
    """
    name_lower = name.lower()

    if weight_strategy == 'mixed':
        # Sparse protection
        if props['sparsity'] > 0.5:
            w_bits = 8
        # First layer (patch embedding) and last layer (head)
        elif layer_idx == 0 or layer_idx == total_layers - 1:
            w_bits = 8
        # Attention layers (Q/K/V projections)
        elif props['layer_type'] == 'attention_qkv' or 'qkv' in name_lower:
            w_bits = 6
        # Projection layers (output of attention)
        elif props['layer_type'] == 'projection' or 'proj' in name_lower:
            w_bits = 6
        # MLP layers (feed-forward) - more robust
        elif 'mlp' in name_lower or 'fc' in name_lower:
            w_bits = 4
        else:
            w_bits = 6
    else:
        w_bits = weight_strategy

    # Activation bits
    if activation_bits == 'mixed':
        if layer_idx == 0 or layer_idx == total_layers - 1:
            a_bits = 8
        elif props['layer_type'] == 'attention_qkv' or 'qkv' in name_lower or 'attn' in name_lower:
            a_bits = 8
        else:
            a_bits = 6
    else:
        a_bits = activation_bits

    return w_bits, a_bits


def assign_bits_swin(layer_idx, total_layers, name, props, weight_strategy, activation_bits):
    """
    Swin Transformer-specific bit assignment.

    Swin characteristics:
    - Window-based attention
    - Patch merging layers (downsampling) are CRITICAL
    - Hierarchical structure with 4 stages
    """
    name_lower = name.lower()

    if weight_strategy == 'mixed':
        # Sparse protection
        if props['sparsity'] > 0.5:
            w_bits = 8
        # Patch embedding and final head
        elif layer_idx == 0 or layer_idx == total_layers - 1:
            w_bits = 8
        # Patch merging layers (CRITICAL for hierarchical structure)
        elif 'downsample' in name_lower or 'patch_merging' in name_lower or 'reduction' in name_lower:
            w_bits = 8
        # Attention layers
        elif props['layer_type'] == 'attention_qkv' or 'qkv' in name_lower or 'attn' in name_lower:
            w_bits = 6
        # Projection layers
        elif 'proj' in name_lower:
            w_bits = 6
        # MLP layers - more robust
        elif 'mlp' in name_lower or 'fc' in name_lower:
            w_bits = 4
        else:
            w_bits = 6
    else:
        w_bits = weight_strategy

    # Activation bits
    if activation_bits == 'mixed':
        if layer_idx == 0 or layer_idx == total_layers - 1:
            a_bits = 8
        elif 'downsample' in name_lower or 'patch_merging' in name_lower:
            a_bits = 8
        elif 'attn' in name_lower or 'qkv' in name_lower:
            a_bits = 8
        else:
            a_bits = 6
    else:
        a_bits = activation_bits

    return w_bits, a_bits


def assign_bits_generic(layer_idx, total_layers, name, props, weight_strategy, activation_bits):
    """
    Generic/conservative bit assignment for unknown architectures.
    """
    if weight_strategy == 'mixed':
        # Very conservative for unknown architectures
        if props['sparsity'] > 0.7:
            w_bits = 8
        elif props['sparsity'] > 0.5:
            w_bits = 6
        # Protect first/last 15% of layers
        elif layer_idx < total_layers * 0.15 or layer_idx >= total_layers * 0.85:
            w_bits = 8
        # Sensitive layer types
        elif props['is_sensitive']:
            w_bits = 6
        else:
            w_bits = 6  # Conservative default
    else:
        w_bits = weight_strategy

    # Conservative activations
    if activation_bits == 'mixed':
        if layer_idx < total_layers * 0.15 or layer_idx >= total_layers * 0.85:
            a_bits = 8
        else:
            a_bits = 6
    else:
        a_bits = activation_bits

    return w_bits, a_bits


# ============================================================================
# Configuration Generation Functions
# ============================================================================

def create_uniform_config(model, weight_bits=8, activation_bits=8):
    """
    Create configuration with uniform bit-widths for all quantizable layers.

    Args:
        model: PyTorch model
        weight_bits: Bit-width for all weights (2, 4, 6, 8, or 32 for FP32)
        activation_bits: Bit-width for all activations (2, 4, 6, 8, or 32 for FP32)

    Returns:
        config: {layer_name: {"weight": bits, "activation": bits}}
    """
    config = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                config[name] = {
                    'weight': weight_bits,
                    'activation': activation_bits
                }
    return config


def create_mixed_weight_config(model, model_name, activation_bits=8):
    """
    Create architecture-aware configuration with mixed-precision weights and uniform activations.

    Args:
        model: PyTorch model
        model_name: Model architecture name (for architecture detection)
        activation_bits: Uniform bit-width for all activations

    Returns:
        config: {layer_name: {"weight": bits, "activation": bits}}
    """
    config = {}
    arch_type = detect_architecture(model, model_name)

    # Collect all quantizable layers with properties
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                props = analyze_layer_properties(module, name)
                layers.append({'name': name, 'module': module, 'properties': props})

    total_layers = len(layers)

    # Architecture-specific bit assignment
    for idx, layer_info in enumerate(layers):
        name = layer_info['name']
        props = layer_info['properties']

        if arch_type == 'vgg':
            w_bits, _ = assign_bits_vgg(idx, total_layers, name, props, 'mixed', activation_bits)
        elif arch_type == 'resnet':
            w_bits, _ = assign_bits_resnet(idx, total_layers, name, props, 'mixed', activation_bits)
        elif arch_type == 'levit':
            w_bits, _ = assign_bits_levit(idx, total_layers, name, props, 'mixed', activation_bits)
        elif arch_type == 'swin':
            w_bits, _ = assign_bits_swin(idx, total_layers, name, props, 'mixed', activation_bits)
        else:
            w_bits, _ = assign_bits_generic(idx, total_layers, name, props, 'mixed', activation_bits)

        config[name] = {'weight': w_bits, 'activation': activation_bits}

    return config


def create_mixed_activation_config(model, model_name, weight_bits=8):
    """
    Create architecture-aware configuration with uniform weights and mixed-precision activations.

    Args:
        model: PyTorch model
        model_name: Model architecture name (for architecture detection)
        weight_bits: Uniform bit-width for all weights

    Returns:
        config: {layer_name: {"weight": bits, "activation": bits}}
    """
    config = {}
    arch_type = detect_architecture(model, model_name)

    # Collect all quantizable layers with properties
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                props = analyze_layer_properties(module, name)
                layers.append({'name': name, 'module': module, 'properties': props})

    total_layers = len(layers)

    # Architecture-specific bit assignment
    for idx, layer_info in enumerate(layers):
        name = layer_info['name']
        props = layer_info['properties']

        if arch_type == 'vgg':
            _, a_bits = assign_bits_vgg(idx, total_layers, name, props, weight_bits, 'mixed')
        elif arch_type == 'resnet':
            _, a_bits = assign_bits_resnet(idx, total_layers, name, props, weight_bits, 'mixed')
        elif arch_type == 'levit':
            _, a_bits = assign_bits_levit(idx, total_layers, name, props, weight_bits, 'mixed')
        elif arch_type == 'swin':
            _, a_bits = assign_bits_swin(idx, total_layers, name, props, weight_bits, 'mixed')
        else:
            _, a_bits = assign_bits_generic(idx, total_layers, name, props, weight_bits, 'mixed')

        config[name] = {'weight': weight_bits, 'activation': a_bits}

    return config


def create_fully_mixed_config(model, model_name):
    """
    Create architecture-aware configuration with mixed-precision for both weights and activations.

    Args:
        model: PyTorch model
        model_name: Model architecture name (for architecture detection)

    Returns:
        config: {layer_name: {"weight": bits, "activation": bits}}
    """
    config = {}
    arch_type = detect_architecture(model, model_name)

    # Collect all quantizable layers with properties
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                props = analyze_layer_properties(module, name)
                layers.append({'name': name, 'module': module, 'properties': props})

    total_layers = len(layers)

    # Architecture-specific bit assignment
    for idx, layer_info in enumerate(layers):
        name = layer_info['name']
        props = layer_info['properties']

        if arch_type == 'vgg':
            w_bits, a_bits = assign_bits_vgg(idx, total_layers, name, props, 'mixed', 'mixed')
        elif arch_type == 'resnet':
            w_bits, a_bits = assign_bits_resnet(idx, total_layers, name, props, 'mixed', 'mixed')
        elif arch_type == 'levit':
            w_bits, a_bits = assign_bits_levit(idx, total_layers, name, props, 'mixed', 'mixed')
        elif arch_type == 'swin':
            w_bits, a_bits = assign_bits_swin(idx, total_layers, name, props, 'mixed', 'mixed')
        else:
            w_bits, a_bits = assign_bits_generic(idx, total_layers, name, props, 'mixed', 'mixed')

        config[name] = {'weight': w_bits, 'activation': a_bits}

    return config


def define_all_experiments(model, model_name, skip_low_bit=False):
    """
    Define all experiments for the comprehensive ablation study.

    Args:
        model: PyTorch model (for config generation)
        model_name: Model architecture name (for architecture-aware mixed-precision)
        skip_low_bit: If True, skip W2 and A2 experiments

    Returns:
        List of experiment dictionaries with 'name', 'category', 'config', 'description'
    """
    experiments = []

    # Detect architecture for logging
    arch_type = detect_architecture(model, model_name)
    print(f"  Detected architecture: {arch_type}")

    # Bit-width options
    weight_bits_options = [8, 6, 4, 2]
    activation_bits_options = [8, 6, 4, 2]

    if skip_low_bit:
        weight_bits_options = [8, 6, 4]
        activation_bits_options = [8, 6, 4]

    # Part 1: Uniform Quantization Grid
    for w_bits in weight_bits_options:
        for a_bits in activation_bits_options:
            experiments.append({
                'name': f'W{w_bits}/A{a_bits}',
                'category': 'uniform',
                'config': create_uniform_config(model, w_bits, a_bits),
                'description': f'Uniform {w_bits}-bit weights, {a_bits}-bit activations'
            })

    # Part 2: Mixed Weights + Uniform Activations (architecture-aware)
    for a_bits in activation_bits_options:
        experiments.append({
            'name': f'W-Mixed/A{a_bits}',
            'category': 'mixed_weights',
            'config': create_mixed_weight_config(model, model_name, activation_bits=a_bits),
            'description': f'Architecture-aware mixed weights, uniform {a_bits}-bit activations'
        })

    # Part 3: Uniform Weights + Mixed Activations (architecture-aware)
    uniform_w_for_mixed_a = [8, 6, 4] if not skip_low_bit else [8, 6, 4]
    for w_bits in uniform_w_for_mixed_a:
        experiments.append({
            'name': f'W{w_bits}/A-Mixed',
            'category': 'mixed_activations',
            'config': create_mixed_activation_config(model, model_name, weight_bits=w_bits),
            'description': f'Uniform {w_bits}-bit weights, architecture-aware mixed activations'
        })

    # Part 4: Fully Mixed (architecture-aware)
    experiments.append({
        'name': 'W-Mixed/A-Mixed',
        'category': 'fully_mixed',
        'config': create_fully_mixed_config(model, model_name),
        'description': 'Architecture-aware mixed-precision for both weights and activations'
    })

    return experiments


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_experiment(model_name, checkpoint_path, config, experiment_name,
                          category, train_loader, test_loader, device='cuda',
                          num_classes=10, input_size=32, calibration_batches=50):
    """
    Run a single quantization experiment with ACTUAL W+A quantization.

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to checkpoint
        config: Quantization configuration {layer: {"weight": bits, "activation": bits}}
        experiment_name: Name of this configuration
        category: Experiment category ('uniform', 'mixed_weights', etc.)
        train_loader: Training data for calibration
        test_loader: Test data for evaluation
        device: Device to use
        num_classes: Number of classes
        input_size: Input image size for BOPs calculation
        calibration_batches: Number of batches for activation calibration

    Returns:
        Results dict
    """
    print(f"\n{'─'*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'─'*60}")
    start_time = time.time()

    # Load fresh model
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    # Apply mixed-precision quantization (both W and A)
    validator = MixedPrecisionValidator(model, config, device)
    validator.apply_weight_quantization()
    validator.setup_activation_quantizers()

    # Calibrate activation quantizers
    calib_time = 0
    if validator.activation_quantizers:
        calib_start = time.time()
        validator.calibrate(train_loader, num_batches=calibration_batches)
        calib_time = time.time() - calib_start

    # Evaluate with error handling for model collapse
    eval_start = time.time()
    try:
        acc = evaluate_accuracy(model, test_loader, device=device)
    except RuntimeError as e:
        if "overflow" in str(e).lower() or "nan" in str(e).lower():
            print(f"  WARNING: Model collapsed (likely too aggressive quantization)")
            acc = 0.0
        else:
            raise
    eval_time = time.time() - eval_start

    # Cleanup hooks
    validator.remove_hooks()

    # Calculate BOPs
    bops = calculate_bops(model, config, input_size)

    # Compute statistics
    w_bits_list = []
    a_bits_list = []
    for layer_name, cfg in config.items():
        if isinstance(cfg, dict):
            w_bits_list.append(cfg.get('weight', 8))
            a_bits_list.append(cfg.get('activation', 8))
        else:
            w_bits_list.append(cfg)
            a_bits_list.append(8)

    avg_w_bits = sum(w_bits_list) / len(w_bits_list) if w_bits_list else 8
    avg_a_bits = sum(a_bits_list) / len(a_bits_list) if a_bits_list else 8

    elapsed = time.time() - start_time

    results = {
        'name': experiment_name,
        'category': category,
        'accuracy': round(acc, 2),
        'avg_weight_bits': round(avg_w_bits, 2),
        'avg_activation_bits': round(avg_a_bits, 2),
        'bops_gbops': round(bops, 2),
        'calibration_time': round(calib_time, 2),
        'evaluation_time': round(eval_time, 2),
        'total_time': round(elapsed, 2)
    }

    print(f"  Accuracy: {acc:.2f}%")
    print(f"  Avg bits: W={avg_w_bits:.1f}, A={avg_a_bits:.1f}")
    print(f"  BOPs: {bops:.2f} GBOPs")
    print(f"  Time: {elapsed:.2f}s")

    # Check for potential model collapse
    if num_classes == 10 and acc < 15.0:
        print(f"  WARNING: Possible model collapse detected (acc < random)")
    elif num_classes == 100 and acc < 2.0:
        print(f"  WARNING: Possible model collapse detected (acc < random)")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Checkpoint Support
# ============================================================================

def save_checkpoint(output_dir, results, completed_experiments):
    """Save intermediate results for crash recovery."""
    checkpoint_path = os.path.join(output_dir, 'ablation_checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'results': results,
            'completed': completed_experiments
        }, f, indent=2)


def load_checkpoint(output_dir):
    """Load checkpoint if exists, return (results, completed_set)."""
    checkpoint_path = os.path.join(output_dir, 'ablation_checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        completed = set(data.get('completed', []))
        return data.get('results', []), completed
    return [], set()


# ============================================================================
# Main Ablation Study Runner
# ============================================================================

def run_comprehensive_ablation_study(model_name, checkpoint_path, dataset='cifar10',
                                      output_dir='ablation_results', calibration_batches=50,
                                      skip_low_bit=False, resume=False):
    """
    Run comprehensive ablation study with all W x A combinations.

    Experiments:
    - 16 uniform combinations (W8/6/4/2 x A8/6/4/2)
    - 4 mixed-weights with uniform activations
    - 3 uniform-weights with mixed activations
    - 1 fully mixed
    - 1 FP32 baseline
    Total: 25 experiments (or fewer if skip_low_bit=True)

    Args:
        model_name: Model architecture name
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name ('cifar10', 'cifar100', 'gtsrb')
        output_dir: Directory for output files
        calibration_batches: Number of batches for activation calibration
        skip_low_bit: If True, skip W2 and A2 experiments
        resume: If True, resume from checkpoint

    Returns:
        Complete ablation results dictionary
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ABLATION STUDY: Weight x Activation Quantization")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}/")
    print(f"Skip low-bit (W2/A2): {skip_low_bit}")
    print(f"{'='*70}\n")

    overall_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Dataset-specific settings
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    if dataset == 'gtsrb':
        input_size = 224
    elif model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    else:
        input_size = 224

    # Batch size
    if model_name == 'swin':
        batch_size = 16
    elif model_name == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load data
    print("Loading data...")
    if dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=batch_size, train=True, input_size=input_size)
        test_loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Load model for configuration generation
    print("Loading model for configuration generation...")
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)

    # Define all experiments (architecture-aware)
    all_experiments = define_all_experiments(model, model_name, skip_low_bit=skip_low_bit)

    # Load checkpoint if resuming
    results = []
    completed_experiments = set()
    if resume:
        results, completed_experiments = load_checkpoint(output_dir)
        if completed_experiments:
            print(f"Resuming from checkpoint: {len(completed_experiments)} experiments completed")

    del model

    # ═══════════════════════════════════════════════════════════════════════
    # FP32 BASELINE
    # ═══════════════════════════════════════════════════════════════════════
    baseline_acc = None
    baseline_bops = None

    if 'FP32_Baseline' not in completed_experiments:
        print(f"\n{'='*60}")
        print("BASELINE: FP32 (No Quantization)")
        print(f"{'='*60}")

        baseline_model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        baseline_start = time.time()
        baseline_acc = evaluate_accuracy(baseline_model, test_loader, device=device)

        # Create FP32 config for BOPs calculation
        config_fp32 = create_uniform_config(baseline_model, weight_bits=32, activation_bits=32)
        baseline_bops = calculate_bops(baseline_model, config_fp32, input_size)
        baseline_time = time.time() - baseline_start

        print(f"  Accuracy: {baseline_acc:.2f}%")
        print(f"  BOPs: {baseline_bops:.2f} GBOPs")
        print(f"  Time: {baseline_time:.2f}s")

        baseline_result = {
            'name': 'FP32_Baseline',
            'category': 'baseline',
            'accuracy': round(baseline_acc, 2),
            'avg_weight_bits': 32.0,
            'avg_activation_bits': 32.0,
            'bops_gbops': round(baseline_bops, 2),
            'calibration_time': 0.0,
            'evaluation_time': round(baseline_time, 2),
            'total_time': round(baseline_time, 2)
        }
        results.append(baseline_result)
        completed_experiments.add('FP32_Baseline')
        save_checkpoint(output_dir, results, list(completed_experiments))

        del baseline_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        # Get baseline from previous results
        for r in results:
            if r['name'] == 'FP32_Baseline':
                baseline_acc = r['accuracy']
                baseline_bops = r['bops_gbops']
                break

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════════
    total_experiments = len(all_experiments) + 1  # +1 for baseline

    for idx, experiment in enumerate(all_experiments):
        exp_name = experiment['name']

        if exp_name in completed_experiments:
            print(f"\n[{idx+2}/{total_experiments}] Skipping (already completed): {exp_name}")
            continue

        print(f"\n[{idx+2}/{total_experiments}] Running: {exp_name}")
        print(f"Category: {experiment['category']}")

        result = run_single_experiment(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=experiment['config'],
            experiment_name=exp_name,
            category=experiment['category'],
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_classes=num_classes,
            input_size=input_size,
            calibration_batches=calibration_batches
        )

        # Add relative metrics
        if baseline_acc is not None:
            result['accuracy_drop'] = round(baseline_acc - result['accuracy'], 2)
        if baseline_bops is not None and result['bops_gbops'] > 0:
            result['bops_reduction'] = round(baseline_bops / result['bops_gbops'], 2)
        else:
            result['bops_reduction'] = 1.0

        results.append(result)
        completed_experiments.add(exp_name)

        # Save checkpoint after each experiment
        save_checkpoint(output_dir, results, list(completed_experiments))

    # ═══════════════════════════════════════════════════════════════════════
    # COMPILE FINAL RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - overall_start

    # Organize by category
    experiments_by_category = {
        'baseline': [],
        'uniform': [],
        'mixed_weights': [],
        'mixed_activations': [],
        'fully_mixed': []
    }
    for r in results:
        cat = r.get('category', 'unknown')
        if cat in experiments_by_category:
            experiments_by_category[cat].append(r)

    final_results = {
        'study_info': {
            'model': model_name,
            'dataset': dataset,
            'device': device,
            'input_size': input_size,
            'calibration_batches': calibration_batches,
            'skip_low_bit': skip_low_bit,
            'total_experiments': len(results),
            'total_time_seconds': round(total_time, 2)
        },
        'baseline': {
            'accuracy': baseline_acc,
            'bops_gbops': baseline_bops
        },
        'experiments_by_category': experiments_by_category,
        'experiments_flat': results
    }

    # Save final results
    results_path = os.path.join(output_dir, 'comprehensive_ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("COMPREHENSIVE ABLATION STUDY RESULTS")
    print(f"{'='*90}")
    print(f"Baseline FP32: {baseline_acc:.2f}% | {baseline_bops:.2f} GBOPs\n")

    # Table 1: Uniform Quantization Grid (4x4)
    print("--- UNIFORM QUANTIZATION GRID ---")
    bit_options = [8, 6, 4, 2] if not skip_low_bit else [8, 6, 4]
    header = f"{'':>8} |" + "".join([f" A{b:>6}% |" for b in bit_options])
    print(header)
    print("-" * len(header))

    uniform_results = {r['name']: r for r in results if r['category'] == 'uniform'}
    for w_bits in bit_options:
        row = f"W{w_bits:>6} |"
        for a_bits in bit_options:
            name = f"W{w_bits}/A{a_bits}"
            if name in uniform_results:
                acc = uniform_results[name]['accuracy']
                row += f" {acc:>7.2f} |"
            else:
                row += f" {'N/A':>7} |"
        print(row)

    # Table 2: All experiments summary
    print(f"\n--- ALL EXPERIMENTS SUMMARY ---")
    print(f"{'Experiment':<25} {'Category':<18} {'Acc':>8} {'Drop':>8} {'W-bits':>8} {'A-bits':>8} {'BOPs':>10} {'Reduce':>8}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: (x['category'], x['name'])):
        drop = r.get('accuracy_drop', 0.0)
        reduction = r.get('bops_reduction', 1.0)
        print(f"{r['name']:<25} {r['category']:<18} {r['accuracy']:>7.2f}% {drop:>7.2f}% "
              f"{r['avg_weight_bits']:>7.1f} {r['avg_activation_bits']:>7.1f} "
              f"{r['bops_gbops']:>9.2f} {reduction:>7.1f}x")

    print(f"\n{'='*90}")
    print(f"Total experiments: {len(results)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {results_path}")
    print(f"{'='*90}\n")

    return final_results


# ============================================================================
# Legacy Function (for backward compatibility)
# ============================================================================

def run_ablation_study(model_name, checkpoint_path, dataset='cifar10',
                       output_dir='ablation_results', calibration_batches=50):
    """
    Legacy wrapper for backward compatibility.
    Calls the comprehensive ablation study.
    """
    return run_comprehensive_ablation_study(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        output_dir=output_dir,
        calibration_batches=calibration_batches,
        skip_low_bit=False,
        resume=False
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Comprehensive Ablation Study for W x A Quantization'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help='Output directory for results')
    parser.add_argument('--calibration-batches', type=int, default=50,
                        help='Number of batches for activation calibration')
    parser.add_argument('--skip-low-bit', action='store_true',
                        help='Skip W2 and A2 experiments to save time')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')

    args = parser.parse_args()

    run_comprehensive_ablation_study(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_dir=args.output_dir,
        calibration_batches=args.calibration_batches,
        skip_low_bit=args.skip_low_bit,
        resume=args.resume
    )
