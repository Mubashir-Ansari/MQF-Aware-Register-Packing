#!/usr/bin/env python3
"""
Automated Mixed-Precision Research Workflow
============================================

Runs all 4 steps for comprehensive mixed-precision activation research:
  1. Ablation Study (global W×A patterns)
  2. Weight Sensitivity (per-layer weight sensitivity)
  3. Activation Sensitivity (per-layer activation sensitivity)
  4. Combined Analysis (generate optimal config)
  5. Validation (test the config)

Usage:
    python run_full_mixed_precision_study.py \
        --model vgg11_bn \
        --checkpoint checkpoints/vgg11_bn.pt \
        --dataset cifar10 \
        --output-dir results/ablation_vgg

Options:
    --skip-ablation       Skip ablation study if JSON exists
    --skip-weight-sens    Skip weight sensitivity if CSV exists
    --skip-activation-sens Skip activation sensitivity if CSV exists
    --skip-validation     Skip final validation
    --bits               Bit-widths to test (default: 2 4 6 8)
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {cmd}\n")

    start_time = time.time()

    ret = os.system(cmd)

    elapsed = time.time() - start_time

    if ret != 0:
        print(f"\n❌ FAILED: {description} (exit code {ret})")
        print(f"   Time elapsed: {elapsed/60:.1f} minutes")
        return False
    else:
        print(f"\n✅ COMPLETED: {description}")
        print(f"   Time elapsed: {elapsed/60:.1f} minutes")
        return True


def check_file_exists(filepath, description):
    """Check if file exists."""
    if os.path.exists(filepath):
        print(f"✓ Found existing: {description}")
        print(f"  Path: {filepath}")
        return True
    else:
        print(f"✗ Not found: {description}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated Mixed-Precision Research Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow
  python run_full_mixed_precision_study.py \\
      --model vgg11_bn \\
      --checkpoint checkpoints/vgg11_bn.pt \\
      --dataset cifar10 \\
      --output-dir results/ablation_vgg

  # Skip existing steps (resume)
  python run_full_mixed_precision_study.py \\
      --model levit \\
      --checkpoint models/best_levit.pth \\
      --dataset cifar10 \\
      --output-dir results/ablation_levit \\
      --skip-ablation \\
      --skip-weight-sens \\
      --skip-activation-sens
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'gtsrb'],
                       help='Dataset name')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for all results')

    # Optional arguments
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 6, 8],
                       help='Bit-widths to test (default: 2 4 6 8)')
    parser.add_argument('--target-drop', type=float, default=2.0,
                       help='Target accuracy drop for optimal config (default: 2.0%%)')
    parser.add_argument('--calibration-batches', type=int, default=50,
                       help='Calibration batches for ablation (default: 50)')
    parser.add_argument('--skip-low-bit', action='store_true',
                       help='Skip W2/A2 in ablation study')

    # Skip flags
    parser.add_argument('--skip-ablation', action='store_true',
                       help='Skip ablation study if results exist')
    parser.add_argument('--skip-weight-sens', action='store_true',
                       help='Skip weight sensitivity if CSV exists')
    parser.add_argument('--skip-activation-sens', action='store_true',
                       help='Skip activation sensitivity if CSV exists')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip combined analysis (only run experiments)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip final validation')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    ablation_json = os.path.join(args.output_dir, 'comprehensive_ablation_results.json')
    weight_csv = os.path.join(args.output_dir, f'{args.model}_weight_sensitivity.csv')
    activation_csv = os.path.join(args.output_dir, f'{args.model}_activation_sensitivity.csv')
    config_json = os.path.join(args.output_dir, f'{args.model}_optimal_mixed_config.json')

    # Print workflow plan
    print("\n" + "="*80)
    print("MIXED-PRECISION RESEARCH WORKFLOW")
    print("="*80)
    print(f"Model:         {args.model}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Dataset:       {args.dataset}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Bit-widths:    {args.bits}")
    print(f"Target Drop:   {args.target_drop}%")
    print("="*80)
    print("\nWorkflow Steps:")
    print("  1. Ablation Study          → comprehensive_ablation_results.json")
    print(f"  2. Weight Sensitivity      → {args.model}_weight_sensitivity.csv")
    print(f"  3. Activation Sensitivity  → {args.model}_activation_sensitivity.csv")
    print(f"  4. Combined Analysis       → {args.model}_optimal_mixed_config.json")
    print("  5. Validation              → Accuracy report")
    print("="*80)

    overall_start = time.time()
    steps_completed = []
    steps_skipped = []

    # ========================================================================
    # STEP 1: Ablation Study
    # ========================================================================
    step1_needed = not (args.skip_ablation and check_file_exists(ablation_json, "Ablation results"))

    if step1_needed:
        print("\n" + "█"*80)
        print("STEP 1/5: ABLATION STUDY (Global W×A Patterns)")
        print("█"*80)
        print("Expected time: 2-4 hours")
        print("This tests all uniform and mixed-precision combinations.")

        bits_str = " ".join(map(str, args.bits))
        cmd = f"python quantization_framework/experiments/activation_experiments/run_ablation_study.py " \
              f"--model {args.model} " \
              f"--checkpoint {args.checkpoint} " \
              f"--dataset {args.dataset} " \
              f"--output-dir {args.output_dir} " \
              f"--calibration-batches {args.calibration_batches} " \
              f"--resume"

        if args.skip_low_bit:
            cmd += " --skip-low-bit"

        success = run_command(cmd, "Ablation Study")

        if not success:
            print("\n❌ Ablation study failed. Aborting workflow.")
            exit(1)

        steps_completed.append("Ablation Study")
    else:
        steps_skipped.append("Ablation Study (already exists)")

    # ========================================================================
    # STEP 2: Weight Sensitivity
    # ========================================================================
    step2_needed = not (args.skip_weight_sens and check_file_exists(weight_csv, "Weight sensitivity"))

    if step2_needed:
        print("\n" + "█"*80)
        print("STEP 2/5: WEIGHT SENSITIVITY (Per-Layer Weight Analysis)")
        print("█"*80)
        print("Expected time: 1-2 hours")
        print("This measures how sensitive each layer is to weight quantization.")

        bits_str = " ".join(map(str, args.bits))
        cmd = f"python quantization_framework/experiments/layer_sensitivity.py " \
              f"--model {args.model} " \
              f"--checkpoint {args.checkpoint} " \
              f"--dataset {args.dataset} " \
              f"--output {weight_csv} " \
              f"--bits {bits_str}"

        success = run_command(cmd, "Weight Sensitivity Analysis")

        if not success:
            print("\n❌ Weight sensitivity failed. Aborting workflow.")
            exit(1)

        steps_completed.append("Weight Sensitivity")
    else:
        steps_skipped.append("Weight Sensitivity (already exists)")

    # ========================================================================
    # STEP 3: Activation Sensitivity
    # ========================================================================
    step3_needed = not (args.skip_activation_sens and check_file_exists(activation_csv, "Activation sensitivity"))

    if step3_needed:
        print("\n" + "█"*80)
        print("STEP 3/5: ACTIVATION SENSITIVITY (Per-Layer Activation Analysis)")
        print("█"*80)
        print("Expected time: 1-2 hours")
        print("This measures how sensitive each layer is to activation quantization.")

        bits_str = " ".join(map(str, args.bits))
        cmd = f"python quantization_framework/experiments/activation_sensitivity.py " \
              f"--model {args.model} " \
              f"--checkpoint {args.checkpoint} " \
              f"--dataset {args.dataset} " \
              f"--output {activation_csv} " \
              f"--bits {bits_str}"

        success = run_command(cmd, "Activation Sensitivity Analysis")

        if not success:
            print("\n❌ Activation sensitivity failed. Aborting workflow.")
            exit(1)

        steps_completed.append("Activation Sensitivity")
    else:
        steps_skipped.append("Activation Sensitivity (already exists)")

    # ========================================================================
    # STEP 4: Combined Analysis
    # ========================================================================
    if not args.skip_analysis:
        print("\n" + "█"*80)
        print("STEP 4/5: COMBINED ANALYSIS (Generate Optimal Config)")
        print("█"*80)
        print("Expected time: <1 minute")
        print("This synthesizes all data and generates the optimal mixed-precision config.")

        # Check if all inputs exist
        missing_inputs = []
        if not os.path.exists(ablation_json):
            missing_inputs.append(ablation_json)
        if not os.path.exists(weight_csv):
            missing_inputs.append(weight_csv)
        if not os.path.exists(activation_csv):
            missing_inputs.append(activation_csv)

        if missing_inputs:
            print("\n⚠️  Warning: Missing input files for analysis:")
            for f in missing_inputs:
                print(f"  - {f}")
            print("\nSkipping analysis step. Run the missing steps first.")
        else:
            cmd = f"python quantization_framework/experiments/analyze_combined_results.py " \
                  f"--ablation {ablation_json} " \
                  f"--sensitivity {weight_csv} " \
                  f"--activation-sensitivity {activation_csv} " \
                  f"--output-config {config_json} " \
                  f"--target-drop {args.target_drop}"

            success = run_command(cmd, "Combined Analysis")

            if not success:
                print("\n⚠️  Analysis failed, but continuing...")
            else:
                steps_completed.append("Combined Analysis")
    else:
        steps_skipped.append("Combined Analysis (skipped by user)")

    # ========================================================================
    # STEP 5: Validation
    # ========================================================================
    if not args.skip_validation and os.path.exists(config_json):
        print("\n" + "█"*80)
        print("STEP 5/5: VALIDATION (Test Generated Config)")
        print("█"*80)
        print("Expected time: 2-5 minutes")
        print("This validates the generated config on the actual model.")

        cmd = f"python quantization_framework/experiments/validate_config.py " \
              f"--model {args.model} " \
              f"--checkpoint {args.checkpoint} " \
              f"--dataset {args.dataset} " \
              f"--config {config_json}"

        success = run_command(cmd, "Config Validation")

        if not success:
            print("\n⚠️  Validation failed.")
        else:
            steps_completed.append("Validation")
    elif args.skip_validation:
        steps_skipped.append("Validation (skipped by user)")
    else:
        print("\n⚠️  Skipping validation: config file not found")
        steps_skipped.append("Validation (config not generated)")

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    total_time = time.time() - overall_start

    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")

    print(f"\nSteps completed ({len(steps_completed)}):")
    for step in steps_completed:
        print(f"  ✅ {step}")

    if steps_skipped:
        print(f"\nSteps skipped ({len(steps_skipped)}):")
        for step in steps_skipped:
            print(f"  ⏭️  {step}")

    print(f"\nOutput files in: {args.output_dir}/")
    if os.path.exists(ablation_json):
        print(f"  ✓ {os.path.basename(ablation_json)}")
    if os.path.exists(weight_csv):
        print(f"  ✓ {os.path.basename(weight_csv)}")
    if os.path.exists(activation_csv):
        print(f"  ✓ {os.path.basename(activation_csv)}")
    if os.path.exists(config_json):
        print(f"  ✓ {os.path.basename(config_json)}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the ablation results to see if mixed-A helps:")
    print(f"   cat {ablation_json} | grep 'W8/A-Mixed'")
    print("\n2. Check the generated optimal config:")
    print(f"   cat {config_json}")
    print("\n3. Use the config in your main pipeline:")
    print(f"   python quantization_framework/experiments/validate_config.py \\")
    print(f"       --model {args.model} --config {config_json}")
    print("\n4. If mixed-A shows significant gains, integrate into auto_quantize_engine.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
