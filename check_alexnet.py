import torch
import sys
import os

# Ensure the quantization_framework is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
framework_path = os.path.join(script_dir, 'quantization_framework')
if framework_path not in sys.path:
    sys.path.append(framework_path)

from models.model_loaders import load_model

def check_model():
    # VERY IMPORTANT: Injection for torch.load mismatch
    import models.alexnet
    sys.modules['__main__'].fasion_mnist_alexnet = models.alexnet.AlexNet
    
    ckpt_path = os.path.join(script_dir, 'models', 'qalex-8bit.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    print(f"--- OFFICIAL VM LOADING TEST ---")
    # This uses the updated loader with 'quanto' support and weights_only=False
    model = load_model('alexnet', checkpoint_path=ckpt_path)
    model.eval()

    from evaluation.pipeline import evaluate_accuracy, get_fashionmnist_dataloader
    
    # Standard FashionMNIST stats
    print("\nEvaluating with Standard FashionMNIST Normalization...")
    dataloader = get_fashionmnist_dataloader(batch_size=128, train=False, input_size=227)
    
    # Evaluate full 10,000 samples for solid proof
    acc = evaluate_accuracy(model, dataloader, device='cpu', max_samples=10000)
    print(f"\n============================================================")
    print(f"VERIFIED 8-BIT BASELINE ACCURACY: {acc:.2f}%")
    print(f"============================================================")
    
    # Check weight types
    first_weight = next(model.parameters())
    print(f"First weight dtype: {first_weight.dtype}")

if __name__ == "__main__":
    check_model()
