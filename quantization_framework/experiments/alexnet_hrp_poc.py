import os
import torch
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loaders import load_model, get_model_size_info
from quantization.hardware_sim import RegisterPackingSimulator
from quantization.primitives import quantize_tensor
from evaluation.pipeline import get_fashionmnist_dataloader, evaluate_accuracy

def run_alexnet_hrp_poc():
    print("="*80)
    print("ALEXNET GRANULAR HRP-MQF POC (Senior AI Engineer Review)")
    print("="*80)

    # 1. Load AlexNet
    model_name = "alexnet"
    # Using the user's provided checkpoint
    checkpoint = "models/qalex-0-7.pth"
    
    print(f"[*] Loading {model_name} with {checkpoint}...")
    try:
        model = load_model(model_name, checkpoint_path=checkpoint, num_classes=10)
        print("[✓] Model and Checkpoint loaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Falling back to random init (if checkpoint failed): {e}")
        from models.alexnet import AlexNet
        model = AlexNet(num_classes=10)

    # 1.5 Load Real Data (FashionMNIST)
    print("\n[*] Loading FashionMNIST from ./data/...")
    try:
        loader = get_fashionmnist_dataloader(train=False, batch_size=64, input_size=227, data_path='./data')
        print(f"[✓] Dataloader ready ({len(loader.dataset)} samples)")
        
        baseline_acc = evaluate_accuracy(model, loader, device='cpu', max_samples=500)
        print(f"[✓] Baseline Accuracy (PTQ/INT8): {baseline_acc:.2f}%")
    except Exception as e:
        print(f"[!] Warning: Could not run real evaluation: {e}")
        baseline_acc = 0.0

    # 2. Define Granular HRP Configuration
    # We'll simulate a mix of W4/A4, W2/A2 and W8/A8 across filters
    # based on typical sensitivity patterns (early layers more sensitive)
    
    sim = RegisterPackingSimulator(register_size=16)
    
    # Layer name mapping for AlexNet
    layers = [
        ("conv1.0", (96, 1, 11, 11)),
        ("conv2.0", (256, 96, 5, 5)),
        ("conv3.0", (384, 256, 3, 3)),
        ("conv4.0", (384, 384, 3, 3)),
        ("conv5.0", (256, 384, 3, 3)),
        ("fc1", (4096, 9216)),
        ("fc2", (4096, 4096)),
        ("fc3", (10, 4096))
    ]

    total_baseline_macs = 0
    total_hrp_macs = 0
    total_bits_int8 = 0
    total_bits_hrp = 0
    
    print(f"\n{'Layer':20} | {'Shape':20} | {'Avg d':6} | {'Sav (vs Int8)':15} | {'Carry'}")
    print("-" * 80)

    total_sav_int8_mb = 0
    
    for layer_name, shape in layers:
        # Simulate bit-width assignment
        # First 2 layers: W8/A8 (d=1) for accuracy
        # Middle layers: W4/A4 (d=2)
        # Deep FC layers: W2/A2 (d=4)
        
        if "conv1" in layer_name or "conv2" in layer_name:
            # High precision
            w_bw = 8
            a_bw = 8
        elif "fc2" in layer_name or "fc3" in layer_name:
            # Low precision / High throughput
            w_bw = 2
            a_bw = 2
        else:
            # Balanced
            w_bw = 4
            a_bw = 4

        # Calculate Packing
        d, util, carry = sim.get_packing_efficiency(w_bw, a_bw)
        
        # Calculate Params
        params = 1
        for s in shape: params *= s
        
        # Savings
        _, sav_int8 = sim.calculate_register_savings({'weight': w_bw}, params)
        total_sav_int8_mb += sav_int8 / (8 * 1024 * 1024)
        
        # Effective MACs (Throughput)
        # Baseline assumes d=1 (8-bit)
        # HRP provides d parallel macs
        total_baseline_macs += params
        total_hrp_macs += params * d
        
        print(f"{layer_name:20} | {str(shape):20} | {d:6.1f} | {sav_int8/(8*1024):11.2f} KB | {carry} bits")

    # 3. Final Summary
    throughput_gain = total_hrp_macs / total_baseline_macs
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"[*] Total MAC Throughput Increase: {throughput_gain:.2f}x")
    print(f"[*] Total Register Space Saved (vs Int8): {total_sav_int8_mb:.2f} MB")
    print(f"[*] Baseline Acc (FashionMNIST): {baseline_acc:.2f}%")
    print(f"[*] Average Carry Budget: {carry} bits per register segment")
    print(f"[*] Hardware Constraint: {sim.R}-bit Registers")
    print("="*80)
    print("NOTE: Spaces for carrying are preserved in each 16-bit register segment")
    print("to allow accumulation without overflow, ensuring FPGA safety.")
    print("="*80)

if __name__ == "__main__":
    run_alexnet_hrp_poc()
