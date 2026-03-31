import argparse
import csv
from pathlib import Path

from cgrp_experiments import MODEL_SPECS, _expanded_joint_config
from packed_simulator import simulate_model, print_simulation_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_accuracy_map(coopt_csv):
    accuracy_map = {}
    with open(coopt_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gamma = float(row["Gamma"])
            if abs(gamma - 0.2) < 1e-9:
                accuracy_map[row["Model"]] = float(row["Accuracy"])
    return accuracy_map


def _alexnet_spatial():
    return {
        "features.0": 121,
        "features.3": 25,
        "features.6": 9,
        "features.8": 9,
        "features.10": 9,
        "classifier.1": 1,
        "classifier.4": 1,
        "classifier.6": 1,
        "conv1.0": 121,
        "conv2.0": 25,
        "conv3.0": 9,
        "conv4.0": 9,
        "conv5.0": 9,
        "fc1": 1,
        "fc2": 1,
        "fc3": 1,
    }


def _vgg_spatial(joint_config):
    return {name: (1 if "classifier" in name else 9) for name in joint_config.keys()}


def _resnet18_spatial(joint_config):
    spatial = {}
    for layer_name in joint_config.keys():
        if "fc" in layer_name or "linear" in layer_name:
            spatial[layer_name] = 1
        elif layer_name == "conv1":
            spatial[layer_name] = 49
        else:
            spatial[layer_name] = 9
    return spatial


def _report_name(label):
    return label.replace("-", "").replace(" ", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coopt-csv", default="results/coopt_ablation.csv")
    args = parser.parse_args()

    accuracy_map = _load_accuracy_map(REPO_ROOT / args.coopt_csv)
    config_paths = {
        "AlexNet": "results/alexnet_gamma_0_2_config.json",
        "VGG-11-BN": "results/vgg11_bn_gamma_0_2_config.json",
        "ResNet-18": "results/resnet18_gamma_0_2_config.json",
    }

    for label, spec in MODEL_SPECS.items():
        joint_config = _expanded_joint_config(spec, config_paths[label])
        if label == "AlexNet":
            spatial = _alexnet_spatial()
        elif label == "VGG-11-BN":
            spatial = _vgg_spatial(joint_config)
        else:
            spatial = _resnet18_spatial(joint_config)

        sim_result = simulate_model(joint_config, spatial, R=16, acc_width=32)
        report = print_simulation_report(sim_result, accuracy_map[label], model_name=label)
        output_path = REPO_ROOT / "results" / f"{_report_name(label)}_sim.txt"
        output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
