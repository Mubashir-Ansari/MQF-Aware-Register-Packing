import torch
import sys
import os
try:
    from quanto import quantize, freeze, qint8
    HAS_QUANTO = True
except ImportError:
    HAS_QUANTO = False

# Default to looking for 'models' relative to the project root, or use env var
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(PROJECT_ROOT, '../models'))

def get_model_size_info(model, checkpoint_path=None):
    """
    Calculate model size in MB.
    Returns dict with size information.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate memory size (assuming FP32)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Get file size if checkpoint provided
    file_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    return {
        'parameters': total_params,
        'size_mb': size_mb,
        'file_size_mb': file_size_mb
    }


def print_model_size(model, checkpoint_path=None, label="Model"):
    """Print model size information in a clean format."""
    info = get_model_size_info(model, checkpoint_path)
    
    print(f"{'='*60}")
    print(f"{label} Size:")
    print(f"  Parameters: {info['parameters']:,} | Memory: {info['size_mb']:.2f} MB", end="")
    if info['file_size_mb']:
        print(f" | File: {info['file_size_mb']:.2f} MB")
    else:
        print()
    print(f"{'='*60}")

def _patch_legacy_alexnet_modules(model):
    """Backfill modules expected by the current AlexNet forward() for old checkpoints."""
    if not hasattr(model, 'relu_fc1'):
        model.relu_fc1 = torch.nn.ReLU()
    if not hasattr(model, 'dropout1'):
        model.dropout1 = torch.nn.Dropout(0.5)
    if not hasattr(model, 'relu_fc2'):
        model.relu_fc2 = torch.nn.ReLU()
    if not hasattr(model, 'dropout2'):
        model.dropout2 = torch.nn.Dropout(0.5)
    return model

def load_model(model_name, checkpoint_path=None, num_classes=10):
    """
    Load a model by name, optionally loading weights from a checkpoint.
    """
    model = None
    if model_name == 'vgg11_bn':
        from .vgg import vgg11_bn
        model = vgg11_bn(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'qvgg-8bit.pth')
    elif model_name == 'levit':
        from .levit import levit_cifar
        model = levit_cifar(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'best3_levit_model_cifar10.pth')
    elif model_name == 'swin':
        from .swin import swin_tiny_patch4_window7_224
        # Warning: Swin architectures vary greatly. This is a best-effort load.
        model = swin_tiny_patch4_window7_224(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'best_swin_model_cifar_changed.pth')
    elif model_name == 'resnet':
        from .resnet import ResNet18
        model = ResNet18(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'qresnet-8bit.pth')
    elif model_name == 'alexnet':
        from .alexnet import alexnet
        model = alexnet(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'qalex-8bit.pth')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load weights if path provided or default exists
    ckpt_to_load = checkpoint_path if checkpoint_path else default_ckpt
    
    if os.path.exists(ckpt_to_load):
        print(f"Loading checkpoint from {ckpt_to_load}")
        try:
            # Fix for 'fasion_mnist_alexnet' not found in __main__
            if 'alexnet' in ckpt_to_load.lower():
                from . import alexnet as alex_mod
                sys.modules['__main__'].fasion_mnist_alexnet = alex_mod.AlexNet
            
            state_dict = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)

            # Some research checkpoints are serialized full model objects
            # (e.g. quantized AlexNet baselines). In that case, use the
            # checkpoint module directly instead of reconstructing and
            # re-quantizing a fresh instance.
            if isinstance(state_dict, torch.nn.Module):
                print(f"Checkpoint contains full {type(state_dict)} module. Using it directly.")
                model = state_dict
                if model_name == 'alexnet':
                    model = _patch_legacy_alexnet_modules(model)
                print_model_size(model, ckpt_to_load, label=f"{model_name}")
                return model
             
            # Case 1: Full model or object with state_dict
            if hasattr(state_dict, 'state_dict') and not isinstance(state_dict, dict):
                print(f"Checkpoint contains {type(state_dict)} object. Extracting state_dict.")
                state_dict = state_dict.state_dict()
            
            # Case 2: Wrapped in dict
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            else:
                print(f"Warning: Loaded checkpoint type {type(state_dict)} is not a dict or model-like.")
                
            # Remove module. prefix if present (DataParallel)
            # Also remove model. prefix (common in some training wrappers)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                if name.startswith('model.'):
                    name = name[6:]
                new_state_dict[name] = v
                
            # Handle quanto quantized checkpoints (e.g. 8-bit AlexNet baseline)
            if model_name == 'alexnet' and HAS_QUANTO:
                print("Applying 'quanto' quantization (Weights+Activations) to model before loading state_dict...")
                # The checkpoint contains both weight scales and activation scales
                quantize(model, weights=qint8, activations=qint8) 
            
            # Strict=False to allow for minor architecture mismatches during research dev
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            if model_name == 'alexnet' and HAS_QUANTO:
                print("Freezing 'quanto' model to finalize weights...")
                freeze(model)
                
            print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print(f"Checkpoint not found at {ckpt_to_load}, returning random init model.")
        
    print_model_size(model, ckpt_to_load, label=f"{model_name}")
    
    return model
