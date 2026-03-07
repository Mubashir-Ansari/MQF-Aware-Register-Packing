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
    
    ckpt_path = os.path.join(script_dir, 'models', 'qalex-0-7.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    print(f"--- OFFICIAL LOADING TEST ---")
    # This uses the updated loader with 'quanto' support
    model = load_model('alexnet', checkpoint_path=ckpt_path)
    model.eval()

    # Normalization Hunt
    norm_schemes = [
        ("No Norm (0..1)", None),
        ("Standard (0.5, 0.5)", (0.5, 0.5)),
        ("FashionMNIST (0.2860, 0.3530)", (0.2860, 0.3530)),
        ("MNIST-style (0.1307, 0.3081)", (0.1307, 0.3081))
    ]

    from evaluation.pipeline import get_fashionmnist_dataloader, evaluate_accuracy
    from torchvision import transforms, datasets

    for name, stats in norm_schemes:
        print(f"\nTesting: {name}")
        
        # Override transforms manually to be sure
        t_list = [transforms.Resize((227, 227)), transforms.ToTensor()]
        if stats:
            t_list.append(transforms.Normalize((stats[0],), (stats[1],)))
        transform = transforms.Compose(t_list)
        
        dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        
        acc = evaluate_accuracy(model, loader, device='cpu', max_samples=2000)
        print(f"  Accuracy: {acc:.2f}%")

    # Check weight types
    first_weight = next(model.parameters())
    print(f"\nFirst weight dtype: {first_weight.dtype}")

if __name__ == "__main__":
    check_model()
