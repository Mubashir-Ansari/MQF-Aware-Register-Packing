"""
cgrp.py - Channel-Grouping Register Packing

Terminology:
    b_w         : weight bit-width for one channel
    b_a         : activation bit-width (equals b_w when W=A enforced)
    R           : register width in bits (default 16)
    d           : packing depth = channels sharing one register
    bin         : group of channels assigned to one register by CGRP
    fill_rate   : sum(b_w_c) / (n_regs * R)
    wasted_bits : n_regs * R - sum(b_w_c)
    packed_issue: one SIMD instruction covering d parallel MACs
    MAC count   : total multiply-accumulate ops - UNCHANGED by packing

Overflow constraint (conservative heterogeneous form):
    For all channels i in one bin:
        sum_i( (2^b_wi - 1) * (2^b_ai - 1) ) < 2^R
    Guarantees 32-bit accumulator cannot overflow.

CGRP never zero-pads narrow channels to match wide neighbours.
This is the core correctness improvement over post-hoc packing.
"""

from math import ceil
from typing import List, Tuple, Dict, Any


def cgrp_pack_layer(
    channels: List[Tuple[int, int]],
    R: int = 16
) -> Dict[str, Any]:
    """
    Run CGRP bin-packing for one layer.

    Args:
        channels : list of (b_w, b_a) tuples, one per channel.
                   Build from joint config as:
                   list(zip(layer_cfg["weight"], layer_cfg["activation"]))
        R        : register width in bits. Default 16.

    Returns dict:
        bins                    : list of lists of (b_w, b_a) tuples
        n_regs                  : int
        fill_rate               : float - sum(b_w) / (n_regs * R)
        wasted_bits             : int
        packed_issue_reduction  : float - C / n_regs
        zero_padding_waste      : int - always 0 by construction
    """
    if not channels:
        raise ValueError("cgrp_pack_layer: channels list is empty")

    # Sort descending by total operand width - FFD heuristic
    sorted_channels = sorted(channels, key=lambda x: x[0] + x[1], reverse=True)
    bins: List[List[Tuple[int, int]]] = []

    for bw, ba in sorted_channels:
        placed = False
        for b in bins:
            # Constraint A: storage fit
            if sum(x[0] for x in b) + bw > R:
                continue
            # Constraint B: overflow safety (conservative heterogeneous)
            overflow = sum((2**x[0] - 1) * (2**x[1] - 1) for x in b)
            overflow += (2**bw - 1) * (2**ba - 1)
            if overflow >= 2**R:
                continue
            b.append((bw, ba))
            placed = True
            break
        if not placed:
            bins.append([(bw, ba)])

    C = len(channels)
    n_regs = len(bins)
    total_w_bits = sum(sum(x[0] for x in b) for b in bins)
    fill_rate = total_w_bits / (n_regs * R)

    return {
        "bins": bins,
        "n_regs": n_regs,
        "fill_rate": round(fill_rate, 4),
        "wasted_bits": n_regs * R - total_w_bits,
        "packed_issue_reduction": round(C / n_regs, 4),
        "zero_padding_waste": 0,
    }


def cgrp_pack_model(
    joint_config: Dict[str, Dict],
    R: int = 16
) -> Dict[str, Any]:
    """
    Run CGRP across all layers from the joint config JSON.

    Args:
        joint_config : full joint config dict loaded from JSON
        R            : register width in bits

    Returns dict:
        layers                        : dict layer_name -> cgrp_pack_layer result
        total_n_regs                  : int
        total_channels                : int
        global_fill_rate              : float
        total_wasted_bits             : int
        global_packed_issue_reduction : float
    """
    layer_results = {}
    total_n_regs = 0
    total_channels = 0
    total_w_bits = 0
    total_wasted = 0

    for layer_name, layer_cfg in joint_config.items():
        channels = list(zip(layer_cfg["weight"], layer_cfg["activation"]))
        result = cgrp_pack_layer(channels, R)
        layer_results[layer_name] = result
        total_n_regs += result["n_regs"]
        total_channels += len(channels)
        total_w_bits += sum(sum(x[0] for x in b) for b in result["bins"])
        total_wasted += result["wasted_bits"]

    global_fill = total_w_bits / (total_n_regs * R) if total_n_regs > 0 else 0.0

    return {
        "layers": layer_results,
        "total_n_regs": total_n_regs,
        "total_channels": total_channels,
        "global_fill_rate": round(global_fill, 4),
        "total_wasted_bits": total_wasted,
        "global_packed_issue_reduction": round(
            total_channels / total_n_regs, 4
        ) if total_n_regs > 0 else 0.0,
    }


def posthoc_pack_layer(
    channels: List[Tuple[int, int]],
    R: int = 16
) -> Dict[str, Any]:
    """
    Replicates the existing post-hoc packing logic for ablation comparison only.
    Uses max(b_w) across all channels - the WRONG approach for granular configs.
    """
    if not channels:
        raise ValueError("posthoc_pack_layer: channels list is empty")

    max_bw = max(bw for bw, _ in channels)
    d = max(1, R // max_bw)
    C = len(channels)
    n_regs = ceil(C / d)
    total_w_bits = sum(bw for bw, _ in channels)
    fill_rate = total_w_bits / (n_regs * R)
    zero_padding_waste = sum(max_bw - bw for bw, _ in channels)

    return {
        "d": d,
        "n_regs": n_regs,
        "fill_rate": round(fill_rate, 4),
        "wasted_bits": n_regs * R - total_w_bits,
        "packed_issue_reduction": round(C / n_regs, 4),
        "zero_padding_waste": zero_padding_waste,
    }


def packing_score_delta(
    channel_idx: int,
    candidate_bw: int,
    candidate_ba: int,
    current_channels: List[Tuple[int, int]],
    R: int = 16
) -> float:
    """
    Compute change in fill_rate if one channel's bit-width changes.
    Used by the co-optimization scoring function inside the search loop.

    Returns:
        positive float = candidate improves packing
        negative float = candidate hurts packing
        0.0            = no change
    """
    current_fill = cgrp_pack_layer(current_channels, R)["fill_rate"]
    candidate_channels = list(current_channels)
    candidate_channels[channel_idx] = (candidate_bw, candidate_ba)
    candidate_fill = cgrp_pack_layer(candidate_channels, R)["fill_rate"]
    return round(candidate_fill - current_fill, 6)
