"""
packed_simulator.py - FPGA-style packed inference simulator

Software execution accounting tool. Counts instructions, storage words,
and accumulation steps the hardware would produce with CGRP packing.
No numeric computation performed. No real hardware required.

CRITICAL DISTINCTION:
    packed_issue_count = packed SIMD instructions issued
                         one instruction = d parallel MACs
    mac_count          = packed_issue_count * d (per bin)
                       = UNCHANGED by packing
    Never report "packing reduces multiplications" - it reduces
    instruction count only.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

from cgrp import cgrp_pack_layer


@dataclass
class BinSimResult:
    channels: List[Tuple[int, int]]
    lane_count: int
    weight_words: int
    activation_words: int
    slack_bits: int
    packed_mac_calls: int
    mac_count: int
    needs_repacking: bool
    precision_label: str


@dataclass
class LayerSimResult:
    layer_name: str
    n_channels: int
    n_bins: int
    weight_storage_words: int
    activation_storage_words: int
    packed_issue_count: int
    mac_count: int
    repacking_ops: int
    bucket_switches: int
    accumulation_steps: int
    fill_rate: float
    bins: List[BinSimResult] = field(default_factory=list)


def _precision_label(b: List[Tuple[int, int]]) -> str:
    widths = sorted(set(bw for bw, _ in b))
    return f"{widths[0]}b" if len(widths) == 1 else "mixed-" + "-".join(str(w) for w in widths) + "b"


def simulate_layer(
    layer_name: str,
    channels: List[Tuple[int, int]],
    spatial_size: int,
    R: int = 16,
    acc_width: int = 32
) -> LayerSimResult:
    """
    Simulate execution accounting for one layer.

    Args:
        layer_name   : layer name string (for reporting)
        channels     : list(zip(layer_cfg["weight"], layer_cfg["activation"]))
        spatial_size : kernel_h * kernel_w
                       FC layer  -> 1
                       conv3x3   -> 9
                       conv5x5   -> 25
                       conv1x1   -> 1
                       conv7x7   -> 49
        R            : register width (default 16)
        acc_width    : accumulator width (default 32)
    """
    pack = cgrp_pack_layer(channels, R)
    bins_raw = pack["bins"]

    bin_results = []
    tot_w_words = tot_a_words = tot_packed = tot_macs = 0
    tot_repack = tot_accum = bucket_sw = 0
    prev_label = None

    for b in bins_raw:
        lane_count = len(b)
        used_bits = sum(x[0] for x in b)
        slack_bits = R - used_bits
        label = _precision_label(b)

        w_words = spatial_size
        a_words = spatial_size

        packed_calls = spatial_size
        mac_count = spatial_size * lane_count

        needs_repack = len(set(x[0] for x in b)) > 1
        if needs_repack:
            tot_repack += 1

        if prev_label is not None and label != prev_label:
            bucket_sw += 1
        prev_label = label

        tot_w_words += w_words
        tot_a_words += a_words
        tot_packed += packed_calls
        tot_macs += mac_count
        tot_accum += mac_count

        bin_results.append(BinSimResult(
            channels=b,
            lane_count=lane_count,
            weight_words=w_words,
            activation_words=a_words,
            slack_bits=slack_bits,
            packed_mac_calls=packed_calls,
            mac_count=mac_count,
            needs_repacking=needs_repack,
            precision_label=label,
        ))

    return LayerSimResult(
        layer_name=layer_name,
        n_channels=len(channels),
        n_bins=pack["n_regs"],
        weight_storage_words=tot_w_words,
        activation_storage_words=tot_a_words,
        packed_issue_count=tot_packed,
        mac_count=tot_macs,
        repacking_ops=tot_repack,
        bucket_switches=bucket_sw,
        accumulation_steps=tot_accum,
        fill_rate=pack["fill_rate"],
        bins=bin_results,
    )


def simulate_model(
    joint_config: Dict[str, Dict],
    layer_spatial_sizes: Dict[str, int],
    R: int = 16,
    acc_width: int = 32
) -> Dict[str, Any]:
    """
    Run simulator across all layers.

    Args:
        joint_config        : full joint config dict
        layer_spatial_sizes : layer_name -> spatial_size
                              Missing layers default to 1 (FC assumption)
    """
    layer_results = {}
    totals = dict(
        weight_storage_words=0,
        activation_storage_words=0,
        packed_issue_count=0,
        mac_count=0,
        repacking_ops=0,
        bucket_switches=0,
        accumulation_steps=0,
    )

    for layer_name, layer_cfg in joint_config.items():
        channels = list(zip(layer_cfg["weight"], layer_cfg["activation"]))
        spatial = layer_spatial_sizes.get(layer_name, 1)
        if layer_name not in layer_spatial_sizes:
            print(f"  [WARN] {layer_name} not in spatial_sizes dict, defaulting to 1")
        result = simulate_layer(layer_name, channels, spatial, R, acc_width)
        layer_results[layer_name] = result
        for k in totals:
            totals[k] += getattr(result, k)

    return {"layers": layer_results, "totals": totals}


def print_simulation_report(
    sim_result: Dict[str, Any],
    accuracy: float,
    model_name: str = ""
) -> str:
    sep = "=" * 80
    lines = [sep, f"  PACKED INFERENCE SIMULATION REPORT - {model_name}", sep, ""]
    lines += [
        "  METRIC DEFINITIONS",
        "  weight_storage_words    : 16-bit packed registers holding weight values",
        "  activation_storage_words: 16-bit packed registers holding activations",
        "  packed_issue_count      : SIMD packed instructions (one = d MACs)",
        "  mac_count               : actual MACs [UNCHANGED by packing]",
        "  repacking_ops           : bins with mixed-width channels",
        "  bucket_switches         : transitions between precision groups",
        "  accumulation_steps      : partial product additions to accumulator",
        "  fill_rate               : weight bits used / total register capacity",
        "",
        "  IMPORTANT: packed_issue_count != mac_count.",
        "  Packing reduces instruction count, NOT arithmetic operations.",
        "",
    ]

    hdr = (
        f"{'Layer':<32}{'Bins':>4}{'W-wds':>7}{'A-wds':>7}"
        f"{'P-Issue':>8}{'MACs':>9}{'Repack':>7}{'Sw':>4}{'Fill':>7}"
    )
    lines += ["  PER-LAYER SUMMARY", "  " + "-" * 76, "  " + hdr, "  " + "-" * 76]

    for name, result in sim_result["layers"].items():
        short_name = name[-30:] if len(name) > 30 else name
        lines.append(
            f"  {short_name:<32}{result.n_bins:>4}{result.weight_storage_words:>7}"
            f"{result.activation_storage_words:>7}{result.packed_issue_count:>8}"
            f"{result.mac_count:>9}{result.repacking_ops:>7}{result.bucket_switches:>4}"
            f"{result.fill_rate * 100:>6.1f}%"
        )

    lines += ["  " + "-" * 76, ""]
    totals = sim_result["totals"]
    lines += [
        "  MODEL TOTALS",
        f"  Weight storage words      : {totals['weight_storage_words']:>12,}",
        f"  Activation storage words  : {totals['activation_storage_words']:>12,}",
        f"  Total packed issues       : {totals['packed_issue_count']:>12,}",
        f"  Total MAC count           : {totals['mac_count']:>12,}  [unchanged by packing]",
        f"  Total repacking ops       : {totals['repacking_ops']:>12,}",
        f"  Total bucket switches     : {totals['bucket_switches']:>12,}",
        f"  Total accumulation steps  : {totals['accumulation_steps']:>12,}",
        "",
        f"  Accuracy (quantized PTQ)  : {accuracy:.4f}%",
        sep,
    ]
    report = "\n".join(lines)
    print(report)
    return report
