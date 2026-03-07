"""
Quantization-Aware Training with Mixed-Precision Activations

Extends QAT to support per-layer activation bit-widths in addition
to per-layer weight bit-widths.

Usage:
    python qat_mixed_precision.py \
        --model vgg11_bn \
        --checkpoint path/to/checkpoint.pth \
        --config mixed_config.json \
        --dataset cifar10 \
        --epochs 5 \
        --output qat_mixed_model.pth
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)


class FakeQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for fake quantization.
    Forward: Quantize -> Dequantize (simulates quantization error)
    Backward: Pass gradients straight through (ignore quantization)
    """
    @staticmethod
    def forward(ctx, x, bit_width, symmetric=True):
        if symmetric:
            q_min = -(2 ** (bit_width - 1))
            q_max = 2 ** (bit_width - 1) - 1
            max_abs = torch.max(torch.abs(x))
            scale = max_abs / q_max if q_max > 0 else torch.tensor(1.0, device=x.device)
            scale = scale.clamp(min=1e-8)

            x_int = torch.round(x / scale).clamp(q_min, q_max)
            x_quant = x_int * scale
        else:
            q_min = 0
            q_max = 2 ** bit_width - 1
            min_val = torch.min(x)
            max_val = torch.max(x)
            scale = (max_val - min_val) / (q_max - q_min)
            scale = scale.clamp(min=1e-8)
            zero_point = torch.round(-min_val / scale).clamp(q_min, q_max)

            x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
            x_quant = (x_int - zero_point) * scale

        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient straight through
        return grad_output, None, None


def fake_quantize_ste(x, bit_width, symmetric=True):
    """Wrapper for FakeQuantizeSTE autograd function."""
    return FakeQuantizeSTE.apply(x, bit_width, symmetric)


class ActivationFakeQuantizer(nn.Module):
    """
    Fake quantization module for activations with STE gradient support.
    Used during QAT to simulate quantization noise in activations.
    """
    def __init__(self, bit_width=8, enabled=True):
        super().__init__()
        self.bit_width = bit_width
        self.enabled = enabled

    def forward(self, x):
        if not self.enabled or not self.training:
            # During eval, use standard quantization
            if self.enabled:
                return fake_quantize_ste(x, self.bit_width, symmetric=False)
            return x

        # During training, use STE for gradient flow
        return fake_quantize_ste(x, self.bit_width, symmetric=False)

    def extra_repr(self):
        return f'bit_width={self.bit_width}, enabled={self.enabled}'


class MixedPrecisionQATWrapper(nn.Module):
    """
    Wraps model to apply fake quantization to BOTH weights AND activations
    during training, with per-layer bit-width configuration.

    Config format:
        {layer_name: {"weight": bits, "activation": bits}}
    """
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        # Parse and validate config
        self.weight_config = {}
        self.activation_config = {}

        for layer_name, cfg in config.items():
            if isinstance(cfg, dict):
                if 'weight' in cfg:
                    self.weight_config[layer_name] = cfg['weight']
                if 'activation' in cfg:
                    self.activation_config[layer_name] = cfg['activation']
            elif isinstance(cfg, int):
                # Legacy: just weight bits
                self.weight_config[layer_name] = cfg

        # Setup activation quantizers and hooks
        self.activation_quantizers = nn.ModuleDict()
        self.hooks = []

        for name, module in self.model.named_modules():
            if name in self.activation_config:
                bits = self.activation_config[name]
                quantizer = ActivationFakeQuantizer(bit_width=bits)
                # Use safe key (replace dots with underscores)
                safe_name = name.replace('.', '_')
                self.activation_quantizers[safe_name] = quantizer

                def make_hook(quant):
                    def hook(mod, inp, out):
                        return quant(out)
                    return hook

                handle = module.register_forward_hook(make_hook(quantizer))
                self.hooks.append(handle)

        print(f"Mixed-Precision QAT wrapper initialized:")
        print(f"  Weight layers: {len(self.weight_config)}")
        print(f"  Activation layers: {len(self.activation_quantizers)}")

    def forward(self, x):
        # Store original weights
        original_weights = {}

        # Apply fake quantization to weights
        for name, module in self.model.named_modules():
            if name in self.weight_config and hasattr(module, 'weight') and module.weight is not None:
                bits = self.weight_config[name]
                original_weights[name] = module.weight.data.clone()
                w = module.weight.data
                q_w = fake_quantize_ste(w, bits, symmetric=True)
                module.weight.data = q_w

        # Forward pass (activation hooks apply automatically)
        output = self.model(x)

        # Restore original weights (gradient flows through STE)
        for name, original_w in original_weights.items():
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module.weight.data = original_w

        return output

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def train_mixed_qat(model, config, train_loader, val_loader,
                    epochs=5, lr=1e-4, patience=3, device='cuda',
                    best_model_path='qat_mixed_best.pth'):
    """
    Fine-tune model with Quantization-Aware Training for both weights
    and activations.

    Args:
        model: Base model to fine-tune
        config: Mixed precision config {layer: {"weight": bits, "activation": bits}}
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to use
        best_model_path: Path to save best model

    Returns:
        Fine-tuned model, training history dict
    """
    model = model.to(device)
    model.train()

    # Wrap model with mixed-precision QAT
    qat_model = MixedPrecisionQATWrapper(model, config)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs_no_improve = 0
    epoch_times = []
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'epoch_time': []}

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward with fake quantization (both W and A)
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        scheduler.step()

        # Compute epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_acc = evaluate_accuracy(model, val_loader, device=device)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Store history
        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 2))
        history['val_acc'].append(round(val_acc, 2))
        history['epoch_time'].append(round(epoch_time, 2))

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% "
              f"[Time: {epoch_time:.1f}s]")

        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best: {best_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Remove hooks
    qat_model.remove_hooks()

    # Compute timing summary
    total_training_time = time.time() - training_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # Add timing to history
    history['timing'] = {
        'epochs_completed': len(epoch_times),
        'avg_epoch_seconds': round(avg_epoch_time, 2),
        'total_training_seconds': round(total_training_time, 2),
        'best_val_accuracy': round(best_acc, 2)
    }

    print(f"\n{'='*60}")
    print(f"MIXED-PRECISION QAT COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Val Accuracy:   {best_acc:.2f}%")
    print(f"  Epochs completed:    {len(epoch_times)}")
    print(f"  Avg time per epoch:  {avg_epoch_time:.2f}s")
    print(f"  Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    print(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixed-Precision QAT (Weight + Activation)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to mixed-precision config JSON')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output', type=str, default='qat_mixed_model.pth',
                        help='Output model path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Initialize timing
    overall_start = time.time()

    # Auto-adjust batch size for memory-intensive models
    if args.model == 'swin' and args.batch_size > 16:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 16 for Swin")
        args.batch_size = 16
    elif args.model == 'levit' and args.batch_size > 32:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 32 for LeViT")
        args.batch_size = 32

    # Load config
    print(f"\nLoading config from {args.config}...")
    with open(args.config, 'r') as f:
        config_data = json.load(f)

    # Handle nested config format
    if 'config' in config_data:
        config = config_data['config']
    else:
        config = config_data

    # Determine num_classes
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # Load model
    print(f"Loading {args.model}...")
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)

    # Determine input size
    if args.dataset == 'gtsrb':
        input_size = 224
    elif args.model in ['levit', 'swin']:
        input_size = 224
    else:
        input_size = 32

    # Load data
    print(f"Loading {args.dataset} (Input: {input_size}x{input_size})...")
    if args.dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_cifar100_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_cifar10_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)

    # Run QAT
    print(f"\nStarting Mixed-Precision QAT...")
    best_path = f"{args.model}_qat_mixed_best.pth"
    model, history = train_mixed_qat(
        model, config, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=args.device, best_model_path=best_path
    )

    # Save final model
    torch.save(model.state_dict(), args.output)
    print(f"Saved QAT model to {args.output}")

    # Save training history
    history_path = args.output.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Overall timing
    total_time = time.time() - overall_start
    print(f"\n[TIMING] Total script time: {total_time:.2f}s ({total_time/60:.1f} min)")
