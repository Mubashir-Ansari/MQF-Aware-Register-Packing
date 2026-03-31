import argparse
import csv
import json
import re
import subprocess
import sys
import time
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "quantization_framework") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "quantization_framework"))

from analysis.cgrp import cgrp_pack_layer, cgrp_pack_model, posthoc_pack_layer
from models.model_loaders import load_model


MODEL_SPECS = {
    "AlexNet": {
        "model": "alexnet",
        "checkpoint": "models/qalex-8bit.pth",
        "dataset": "fashionmnist",
        "profile": "alexnet_sensitivity_2_4_8.csv",
        "num_classes": 10,
    },
    "VGG-11-BN": {
        "model": "vgg11_bn",
        "checkpoint": "models/qvgg-8bit.pth",
        "dataset": "cifar10",
        "profile": "vgg11_bn_sensitivity_2_4_8.csv",
        "num_classes": 10,
    },
    "ResNet-18": {
        "model": "resnet18",
        "checkpoint": "models/resnet18.pt",
        "dataset": "cifar10",
        "profile": "resnet18_sensitivity_2_4_8.csv",
        "num_classes": 10,
    },
}


def _run(cmd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed.stdout


def _out_units(model_name, checkpoint_path, num_classes):
    model = load_model(model_name, checkpoint_path=str(REPO_ROOT / checkpoint_path), num_classes=num_classes)
    counts = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            counts[name] = int(module.weight.shape[0])
    return counts


def _load_joint_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["config"] if "config" in data else data


def _expand_layer_cfg(layer_name, layer_cfg, out_units):
    channels = max(1, int(out_units.get(layer_name, 1)))
    weight = layer_cfg["weight"]
    activation = layer_cfg["activation"]
    if isinstance(weight, int):
        weight = [weight] * channels
    if isinstance(activation, int):
        activation = [activation] * channels
    if len(weight) != len(activation):
        raise ValueError(f"Layer {layer_name} has mismatched W/A lengths")
    return {"weight": [int(v) for v in weight], "activation": [int(v) for v in activation]}


def _expanded_joint_config(spec, joint_config_path):
    out_units = _out_units(spec["model"], spec["checkpoint"], spec["num_classes"])
    joint_cfg = _load_joint_config(REPO_ROOT / joint_config_path)
    return {
        layer_name: _expand_layer_cfg(layer_name, layer_cfg, out_units)
        for layer_name, layer_cfg in joint_cfg.items()
    }


def _parse_accuracy(stdout):
    match = re.search(r"Final Mixed-Precision Accuracy:\s*([0-9.]+)%", stdout)
    if not match:
        raise RuntimeError(f"Could not parse accuracy from validate_config output:\n{stdout}")
    return float(match.group(1))


def run_search(spec, gamma, max_samples):
    gamma_tag = str(gamma).replace(".", "_")
    output_name = f"results/{spec['model']}_gamma_{gamma_tag}_config.json"
    start = time.time()
    _run([
        "python",
        "quantization_framework/experiments/joint_search.py",
        "--model", spec["model"],
        "--checkpoint", spec["checkpoint"],
        "--profile", spec["profile"],
        "--dataset", spec["dataset"],
        "--bits", "2", "4", "8",
        "--target-drop", "3.0",
        "--register-size", "16",
        "--storage-gain-weight", "1.0",
        "--score-gamma", str(gamma),
        "--output", output_name,
        "--device", "cpu",
    ])
    search_time = round(time.time() - start, 3)
    base = output_name[:-5]
    validate_stdout = _run([
        "python",
        "quantization_framework/experiments/validate_config.py",
        "--model", spec["model"],
        "--checkpoint", spec["checkpoint"],
        "--config", f"{base}_weight.json",
        "--activation-config", f"{base}_activation.json",
        "--dataset", spec["dataset"],
        "--device", "cpu",
        "--max-samples", str(max_samples),
        "--batch-size", "32",
    ])
    accuracy = _parse_accuracy(validate_stdout)
    expanded_cfg = _expanded_joint_config(spec, output_name)
    cgrp_model = cgrp_pack_model(expanded_cfg, R=16)
    return {
        "output": output_name,
        "accuracy": accuracy,
        "global_fill_rate": cgrp_model["global_fill_rate"],
        "global_pir": cgrp_model["global_packed_issue_reduction"],
        "search_time_s": search_time,
        "config": expanded_cfg,
    }


def write_coopt_ablation(output_csv, max_samples):
    rows = []
    results = {}
    for label, spec in MODEL_SPECS.items():
        results[label] = {}
        for gamma in (0.0, 0.2):
            result = run_search(spec, gamma, max_samples)
            results[label][gamma] = result
            rows.append({
                "Model": label,
                "Gamma": gamma,
                "Accuracy": result["accuracy"],
                "GlobalFillRate": result["global_fill_rate"],
                "GlobalPIR": result["global_pir"],
                "SearchTime_s": result["search_time_s"],
            })
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Model", "Gamma", "Accuracy", "GlobalFillRate", "GlobalPIR", "SearchTime_s"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return results


def write_cgrp_ablation(output_csv, coopt_results):
    rows = []
    for label, gamma_results in coopt_results.items():
        expanded_cfg = gamma_results[0.2]["config"]
        for layer_name, layer_cfg in expanded_cfg.items():
            channels = list(zip(layer_cfg["weight"], layer_cfg["activation"]))
            cgrp = cgrp_pack_layer(channels, R=16)
            posthoc = posthoc_pack_layer(channels, R=16)
            if cgrp["n_regs"] > posthoc["n_regs"]:
                raise RuntimeError(
                    f"[BUG] Layer {layer_name}: CGRP gave MORE registers than post-hoc"
                )
            rows.append({
                "Model": label,
                "Layer": layer_name,
                "PostHoc_nRegs": posthoc["n_regs"],
                "PostHoc_PIR": posthoc["packed_issue_reduction"],
                "CGRP_nRegs": cgrp["n_regs"],
                "CGRP_FillRate": cgrp["fill_rate"],
                "CGRP_PIR": cgrp["packed_issue_reduction"],
                "Delta_nRegs": posthoc["n_regs"] - cgrp["n_regs"],
            })
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Model", "Layer", "PostHoc_nRegs", "PostHoc_PIR", "CGRP_nRegs",
            "CGRP_FillRate", "CGRP_PIR", "Delta_nRegs"
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coopt-output", default="results/coopt_ablation.csv")
    parser.add_argument("--cgrp-output", default="results/cgrp_ablation.csv")
    parser.add_argument("--max-samples", type=int, default=128)
    args = parser.parse_args()

    results = write_coopt_ablation(REPO_ROOT / args.coopt_output, args.max_samples)
    write_cgrp_ablation(REPO_ROOT / args.cgrp_output, results)


if __name__ == "__main__":
    main()
