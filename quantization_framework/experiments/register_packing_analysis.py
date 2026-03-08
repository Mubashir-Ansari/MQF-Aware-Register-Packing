import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.register_packing_optimizer import run_packing_analysis
from models.model_loaders import load_model

# Fix for checkpoint loading mismatch
import models.alexnet

fasion_mnist_alexnet = models.alexnet.AlexNet
sys.modules["__main__"].fasion_mnist_alexnet = models.alexnet.AlexNet


def _load_cfg(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _flatten_cfg(cfg):
    if isinstance(cfg, dict) and "config" in cfg:
        return cfg["config"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Post-MQF register packing strategy analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "gtsrb", "fashionmnist"])
    parser.add_argument("--weight-config", type=str, required=True)
    parser.add_argument("--activation-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="packing_reports")
    parser.add_argument("--register-size", type=int, default=16)
    parser.add_argument("--acc-width", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--default-input-bits", type=int, default=8)
    parser.add_argument(
        "--aligned-policy",
        type=str,
        default="2:2,3:4,4:4,8:8",
        help="Aligned slot policy as comma-separated map, e.g. 2:2,3:4,4:4,8:8",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "gtsrb":
        num_classes = 43
    else:
        num_classes = 10

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PACKING] Using device for shape tracing: {device}")

    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    model = model.to(device).eval()

    w_cfg = _flatten_cfg(_load_cfg(args.weight_config))
    a_cfg = _flatten_cfg(_load_cfg(args.activation_config))
    aligned_policy = {}
    for pair in args.aligned_policy.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        aligned_policy[int(k)] = int(v)

    os.makedirs(args.output_dir, exist_ok=True)
    reports = run_packing_analysis(
        model=model,
        model_name=args.model,
        dataset=args.dataset,
        weight_config=w_cfg,
        activation_config=a_cfg,
        output_dir=args.output_dir,
        register_size=args.register_size,
        acc_width=args.acc_width,
        aligned_policy=aligned_policy,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        default_input_bits=args.default_input_bits,
        device=device,
    )

    # Print compact comparison
    print("\n" + "=" * 78)
    print("PACKING STRATEGY COMPARISON")
    print("=" * 78)
    print(f"{'Strategy':24s} | {'Total Regs':>12s} | {'Savings vs 8b':>12s} | {'Issue Red.':>10s} | {'Cost':>10s}")
    print("-" * 78)
    for name in ["raw_homogeneous", "aligned", "heterogeneous_storage", "hybrid_storage_compute"]:
        r = reports[name]
        print(
            f"{name:24s} | {r.total_registers:12,d} | {r.savings_percent:11.2f}% | "
            f"{r.total_reduction_factor:9.3f}x | {r.objective_cost:10.2f}"
        )
    print("-" * 78)
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    print("  - *_report.json per strategy")
    print("  - per_layer_summary.csv")
    print("  - global_comparison.md")
    print("  - human_report.txt")


if __name__ == "__main__":
    main()
