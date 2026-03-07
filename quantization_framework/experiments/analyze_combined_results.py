"""
Combined Ablation + Sensitivity Analysis
=========================================

Lightweight script to combine and analyze results from:
1. comprehensive_ablation_results.json (from run_ablation_study.py)
2. layer_profile.csv (from layer_sensitivity.py) - weight sensitivity
3. activation_sensitivity.csv (from activation_sensitivity.py) - activation sensitivity [OPTIONAL]

Extracts global patterns (W/A sensitivity) and layer patterns (critical/robust),
then generates an optimal mixed-precision config for BOTH weights AND activations.

Usage:
    # Analysis only (print insights)
    python analyze_combined_results.py \
        --ablation results/comprehensive_ablation_results.json \
        --sensitivity results/vgg11_bn_profile.csv

    # With activation sensitivity (true mixed-precision for W and A)
    python analyze_combined_results.py \
        --ablation results/comprehensive_ablation_results.json \
        --sensitivity results/weight_sensitivity.csv \
        --activation-sensitivity results/activation_sensitivity.csv \
        --output-config optimal_config.json

    # Generate deployment config
    python analyze_combined_results.py \
        --ablation results/comprehensive_ablation_results.json \
        --sensitivity results/vgg11_bn_profile.csv \
        --output-config optimal_config.json \
        --target-drop 2.0
"""

import json
import csv
import argparse
from pathlib import Path


def load_ablation_results(json_path):
    """Load ablation study JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_sensitivity_results(csv_path):
    """
    Load layer sensitivity CSV.

    Returns:
        dict: {layer_name: {bit_width: sensitivity_score}}
    """
    sensitivity = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            sensitivity[layer] = {'baseline_acc': float(row['baseline_acc'])}

            # Extract sensitivity scores for different bit-widths
            for key, value in row.items():
                if key.startswith('sensitivity_') and 'bit' in key:
                    # Parse bit-width from column name (e.g., "sensitivity_4bit" -> 4)
                    bits = int(key.replace('sensitivity_', '').replace('bit', ''))
                    sensitivity[layer][bits] = float(value)

    return sensitivity


def analyze_global_patterns(ablation_results):
    """
    Extract global insights from ablation study.

    Analyzes:
    - Top uniform configurations
    - Activation sensitivity (W=8 fixed, vary A)
    - Weight sensitivity (A=8 fixed, vary W)

    Returns:
        dict: Global insights and recommendations
    """
    baseline = ablation_results['baseline']
    baseline_acc = baseline['accuracy']
    experiments = ablation_results['experiments_flat']

    print("\n" + "="*70)
    print("GLOBAL PATTERNS (from Ablation Study)")
    print("="*70)
    print(f"Baseline: {baseline_acc:.2f}% @ FP32")

    # Get uniform experiments
    uniform_exps = [e for e in experiments if e['category'] == 'uniform']
    uniform_exps.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\nTop 5 Uniform Configurations:")
    for i, exp in enumerate(uniform_exps[:5], 1):
        drop = baseline_acc - exp['accuracy']
        print(f"  {i}. {exp['name']}: {exp['accuracy']:.2f}% (drop: {drop:.2f}%)")

    # Analyze activation sensitivity (fix W=8, vary A)
    print("\nActivation Sensitivity (W=8 fixed):")
    a_sensitivity = {}
    for exp in uniform_exps:
        if exp['avg_weight_bits'] == 8:
            a_bits = int(exp['avg_activation_bits'])
            drop = baseline_acc - exp['accuracy']
            a_sensitivity[a_bits] = drop

            if drop < 5:
                status = "OK"
            elif drop < 20:
                status = "HIGH"
            else:
                status = "COLLAPSE"
            print(f"  A{a_bits}: {exp['accuracy']:.2f}% (drop: {drop:.2f}%) [{status}]")

    # Analyze weight sensitivity (fix A=8, vary W)
    print("\nWeight Sensitivity (A=8 fixed):")
    w_sensitivity = {}
    for exp in uniform_exps:
        if exp['avg_activation_bits'] == 8:
            w_bits = int(exp['avg_weight_bits'])
            drop = baseline_acc - exp['accuracy']
            w_sensitivity[w_bits] = drop

            if drop < 5:
                status = "OK"
            elif drop < 20:
                status = "HIGH"
            else:
                status = "COLLAPSE"
            print(f"  W{w_bits}: {exp['accuracy']:.2f}% (drop: {drop:.2f}%) [{status}]")

    # Determine recommendations
    recommended_a_bits = 8
    for a_bits in sorted(a_sensitivity.keys()):
        if a_sensitivity[a_bits] < 5.0:
            recommended_a_bits = a_bits

    recommended_w_range = [w for w, drop in w_sensitivity.items() if drop < 5.0]
    if not recommended_w_range:
        recommended_w_range = [8]

    # Find minimum activation bits before collapse
    min_a_bits = 8
    for a_bits in sorted(a_sensitivity.keys()):
        if a_sensitivity.get(a_bits, 100) > 20:  # >20% drop = collapse
            break
        min_a_bits = a_bits

    print("\n" + "-"*70)
    print("RECOMMENDATIONS:")
    print(f"  Activation: Use A{min_a_bits}+ (collapse below this)")
    print(f"  Weight: {min(recommended_w_range)}-{max(recommended_w_range)} bit acceptable")
    print("="*70)

    return {
        'baseline_accuracy': baseline_acc,
        'recommended_activation_bits': min_a_bits,
        'recommended_weight_range': sorted(recommended_w_range),
        'activation_sensitivity': a_sensitivity,
        'weight_sensitivity': w_sensitivity,
        'top_configs': [(e['name'], e['accuracy']) for e in uniform_exps[:5]]
    }


def analyze_layer_patterns(sensitivity, label="WEIGHT"):
    """
    Extract per-layer insights from sensitivity analysis.

    Identifies:
    - Critical layers (high sensitivity)
    - Robust layers (low sensitivity)

    Args:
        sensitivity: Dict from load_sensitivity_results()
        label: Label for output (e.g., "WEIGHT" or "ACTIVATION")

    Returns:
        dict: Layer insights with critical/robust classification
    """
    print("\n" + "="*70)
    print(f"{label} LAYER PATTERNS (from Sensitivity Analysis)")
    print("="*70)

    # Determine which bit-widths were tested
    sample_layer = next(iter(sensitivity.values()))
    tested_bits = [k for k in sample_layer.keys() if isinstance(k, int)]

    # Use lowest tested bit-width for ranking (most stressful)
    ranking_bits = min(tested_bits)
    print(f"Ranking by sensitivity at {ranking_bits}-bit\n")

    # Rank layers by sensitivity at lowest bit-width
    layer_ranking = []
    for layer, scores in sensitivity.items():
        if ranking_bits in scores:
            layer_ranking.append((layer, scores[ranking_bits]))

    layer_ranking.sort(key=lambda x: x[1], reverse=True)

    # Identify critical (>10% drop) and robust (<2% drop) layers
    critical_layers = [(l, s) for l, s in layer_ranking if s > 10.0]
    robust_layers = [(l, s) for l, s in layer_ranking if s < 2.0]

    print(f"Critical Layers (>{10}% drop at {ranking_bits}-bit): {len(critical_layers)}")
    for layer, score in critical_layers[:10]:
        print(f"  {layer}: {score:.2f}% drop")
    if len(critical_layers) > 10:
        print(f"  ... and {len(critical_layers) - 10} more")

    print(f"\nRobust Layers (<{2}% drop at {ranking_bits}-bit): {len(robust_layers)}")
    for layer, score in robust_layers[:10]:
        print(f"  {layer}: {score:.2f}% drop")
    if len(robust_layers) > 10:
        print(f"  ... and {len(robust_layers) - 10} more")

    print("="*70)

    return {
        'ranking_bits': ranking_bits,
        'critical_layers': [l for l, _ in critical_layers],
        'robust_layers': [l for l, _ in robust_layers],
        'layer_ranking': layer_ranking,
        'layer_scores': {l: s for l, s in layer_ranking}
    }


def generate_optimal_config(global_insights, weight_insights, activation_insights=None, target_drop=3.0):
    """
    Generate optimal quantization config combining all analyses.

    Strategy:
      - Weight bits: from weight_insights (per-layer)
          * Critical layers: max(recommended_weight_range)
          * Robust layers: min(recommended_weight_range)
          * Others: middle of range
      - Activation bits: from activation_insights if available (per-layer)
          * A-critical layers: highest acceptable A bits
          * A-robust layers: lowest acceptable A bits
          * If no activation_insights: use global recommendation (uniform)

    Args:
        global_insights: From analyze_global_patterns()
        weight_insights: From analyze_layer_patterns() for weights
        activation_insights: From analyze_layer_patterns() for activations (optional)
        target_drop: Target accuracy drop constraint (%)

    Returns:
        config: {layer: {'weight': bits, 'activation': bits}}
    """
    has_activation_data = activation_insights is not None

    print("\n" + "="*70)
    if has_activation_data:
        print(f"GENERATING OPTIMAL CONFIG (MIXED W + MIXED A)")
    else:
        print(f"GENERATING OPTIMAL CONFIG (MIXED W + UNIFORM A)")
    print(f"Target drop: {target_drop}%")
    print("="*70)

    config = {}

    # Weight bit options
    w_range = global_insights['recommended_weight_range']
    w_max = max(w_range)
    w_min = min(w_range)
    w_mid = sorted(w_range)[len(w_range)//2] if len(w_range) > 1 else w_max

    w_critical_layers = set(weight_insights['critical_layers'])
    w_robust_layers = set(weight_insights['robust_layers'])

    # Activation bit options
    a_bits_default = global_insights['recommended_activation_bits']

    if has_activation_data:
        a_critical_layers = set(activation_insights['critical_layers'])
        a_robust_layers = set(activation_insights['robust_layers'])

        # Determine activation bit range based on global analysis
        a_sensitivity = global_insights.get('activation_sensitivity', {})
        a_range = [a for a, drop in a_sensitivity.items() if drop < 20.0]
        if not a_range:
            a_range = [8]
        a_max = max(a_range)
        a_min = min([a for a, drop in a_sensitivity.items() if drop < 5.0]) if a_sensitivity else a_bits_default

        print(f"\nStrategy (MIXED WEIGHTS + MIXED ACTIVATIONS):")
        print(f"  Weight critical ({len(w_critical_layers)} layers): W{w_max}")
        print(f"  Weight robust ({len(w_robust_layers)} layers): W{w_min}")
        print(f"  Activation critical ({len(a_critical_layers)} layers): A{a_max}")
        print(f"  Activation robust ({len(a_robust_layers)} layers): A{a_min}")
    else:
        a_critical_layers = set()
        a_robust_layers = set()
        a_max = a_bits_default
        a_min = a_bits_default

        print(f"\nStrategy (MIXED WEIGHTS + UNIFORM ACTIVATIONS):")
        print(f"  Weight critical ({len(w_critical_layers)} layers): W{w_max}")
        print(f"  Weight robust ({len(w_robust_layers)} layers): W{w_min}")
        print(f"  Activation: A{a_bits_default} everywhere (no per-layer data)")

    # Assign bits per layer
    for layer, _ in weight_insights['layer_ranking']:
        # Weight bits
        if layer in w_critical_layers:
            w_bits = w_max
        elif layer in w_robust_layers:
            w_bits = w_min
        else:
            w_bits = w_mid

        # Activation bits
        if has_activation_data:
            if layer in a_critical_layers:
                a_bits = a_max
            elif layer in a_robust_layers:
                a_bits = a_min
            else:
                a_bits = a_bits_default
        else:
            a_bits = a_bits_default

        config[layer] = {
            'weight': w_bits,
            'activation': a_bits
        }

    # Print summary
    w_dist = {}
    a_dist = {}
    for cfg in config.values():
        w = cfg['weight']
        a = cfg['activation']
        w_dist[w] = w_dist.get(w, 0) + 1
        a_dist[a] = a_dist.get(a, 0) + 1

    print(f"\nGenerated config for {len(config)} layers:")
    print(f"  Weight distribution: {dict(sorted(w_dist.items(), reverse=True))}")
    print(f"  Activation distribution: {dict(sorted(a_dist.items(), reverse=True))}")

    avg_w = sum(cfg['weight'] for cfg in config.values()) / len(config)
    avg_a = sum(cfg['activation'] for cfg in config.values()) / len(config)
    print(f"  Average: W{avg_w:.1f}/A{avg_a:.1f}")

    # Estimate accuracy
    baseline_acc = global_insights['baseline_accuracy']
    estimated_drop = 0
    for layer, cfg in config.items():
        w_bits = cfg['weight']
        layer_sensitivity = weight_insights['layer_scores'].get(layer, 0)
        bit_factor = (8 - w_bits) / 8
        estimated_drop += layer_sensitivity * bit_factor * 0.1

    estimated_acc = baseline_acc - estimated_drop
    print(f"\n  Estimated accuracy: ~{estimated_acc:.1f}% (rough estimate)")
    print("="*70)

    return config


def save_config(config, output_path):
    """
    Save config to JSON (compatible with validate_config.py).

    Removes internal metadata fields before saving.
    """
    # Remove metadata fields for deployment
    deployment_config = {}
    for layer, cfg in config.items():
        deployment_config[layer] = {
            'weight': cfg['weight'],
            'activation': cfg['activation']
        }

    with open(output_path, 'w') as f:
        json.dump(deployment_config, f, indent=2)

    print(f"\nConfig saved to: {output_path}")
    print(f"Use with: python validate_config.py --config {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine ablation + sensitivity analysis for optimal config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just analyze (print insights)
  python analyze_combined_results.py \\
      --ablation results/comprehensive_ablation_results.json \\
      --sensitivity results/vgg11_profile.csv

  # With activation sensitivity (true mixed-precision for W and A)
  python analyze_combined_results.py \\
      --ablation results/comprehensive_ablation_results.json \\
      --sensitivity results/weight_sensitivity.csv \\
      --activation-sensitivity results/activation_sensitivity.csv \\
      --output-config optimal_config.json

  # Generate deployment config
  python analyze_combined_results.py \\
      --ablation results/comprehensive_ablation_results.json \\
      --sensitivity results/vgg11_profile.csv \\
      --output-config optimal_config.json \\
      --target-drop 2.0
        """
    )

    parser.add_argument('--ablation', type=str, required=True,
                       help='Path to comprehensive_ablation_results.json')
    parser.add_argument('--sensitivity', type=str, required=True,
                       help='Path to weight sensitivity CSV')
    parser.add_argument('--activation-sensitivity', type=str, default=None,
                       help='Path to activation sensitivity CSV (optional, enables mixed A)')
    parser.add_argument('--output-config', type=str, default=None,
                       help='Output config JSON path (optional)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                       help='Target accuracy drop %% (default: 3.0)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.ablation).exists():
        print(f"Error: Ablation results not found: {args.ablation}")
        exit(1)

    if not Path(args.sensitivity).exists():
        print(f"Error: Weight sensitivity profile not found: {args.sensitivity}")
        exit(1)

    if args.activation_sensitivity and not Path(args.activation_sensitivity).exists():
        print(f"Error: Activation sensitivity profile not found: {args.activation_sensitivity}")
        exit(1)

    # Load data
    print("Loading data...")
    print(f"  Ablation: {args.ablation}")
    print(f"  Weight Sensitivity: {args.sensitivity}")
    if args.activation_sensitivity:
        print(f"  Activation Sensitivity: {args.activation_sensitivity}")

    ablation_results = load_ablation_results(args.ablation)
    weight_sensitivity = load_sensitivity_results(args.sensitivity)

    activation_sensitivity = None
    if args.activation_sensitivity:
        activation_sensitivity = load_sensitivity_results(args.activation_sensitivity)

    print(f"\nLoaded {len(ablation_results['experiments_flat'])} ablation experiments")
    print(f"Loaded {len(weight_sensitivity)} weight sensitivity profiles")
    if activation_sensitivity:
        print(f"Loaded {len(activation_sensitivity)} activation sensitivity profiles")

    # Analyze
    global_insights = analyze_global_patterns(ablation_results)
    weight_insights = analyze_layer_patterns(weight_sensitivity, label="WEIGHT")

    activation_insights = None
    if activation_sensitivity:
        activation_insights = analyze_layer_patterns(activation_sensitivity, label="ACTIVATION")

    # Generate config if requested
    if args.output_config:
        config = generate_optimal_config(
            global_insights,
            weight_insights,
            activation_insights=activation_insights,
            target_drop=args.target_drop
        )
        save_config(config, args.output_config)
    else:
        print("\nNo --output-config specified.")
        print("Add --output-config <path> to generate deployment config.")


if __name__ == "__main__":
    main()
