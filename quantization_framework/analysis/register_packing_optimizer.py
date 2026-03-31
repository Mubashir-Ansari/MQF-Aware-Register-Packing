import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .cgrp import cgrp_pack_layer, posthoc_pack_layer
except ImportError:
    from cgrp import cgrp_pack_layer, posthoc_pack_layer


@dataclass
class PackedField:
    tensor_kind: str  # weight | activation | output
    layer_name: str
    group_id: str
    original_index: int
    bit_width: int
    aligned_bit_width: int
    register_id: int
    bit_offset: int
    signed: bool


@dataclass
class PackedRegister:
    register_id: int
    register_width: int
    strategy: str
    used_bits: int
    slack_bits: int
    fields: List[PackedField] = field(default_factory=list)


@dataclass
class LayerPackingReport:
    layer_name: str
    layer_type: str
    strategy: str
    num_parameters: int
    weight_bit_distribution: Dict[str, int]
    activation_bit_distribution: Dict[str, int]
    output_bit_distribution: Dict[str, int]
    average_bit_width: float
    logical_storage_bits_weight: int
    logical_storage_bits_activation: int
    logical_storage_bits_output: int
    total_logical_storage_bits: int
    weight_registers: int
    activation_registers: int
    output_registers: int
    total_operand_registers: int
    accumulator_registers_estimate: int
    utilization: float
    avg_register_utilization: float
    slack_bits: int
    scalar_macs: int
    packed_issues: int
    reduction_factor: float
    packed_issue_reduction_factor: float
    lane_utilization: float
    effective_throughput_factor: float
    max_safe_tile_size: int
    overflow_risk: bool
    requires_accumulator_tiling: bool
    storage_words_weight: int
    storage_words_activation: int
    storage_words_output: int
    example_weight_registers: List[Dict[str, object]]
    example_activation_registers: List[Dict[str, object]]
    example_output_registers: List[Dict[str, object]]
    notes: str


@dataclass
class GlobalPackingReport:
    strategy_name: str
    baseline_registers: int
    total_registers: int
    total_storage_words: int
    total_logical_storage_bits: int
    total_slack_bits: int
    total_scalar_macs: int
    total_packed_issues: int
    total_reduction_factor: float
    total_effective_throughput_factor: float
    savings_percent: float
    objective_cost: float
    per_layer_reports: List[LayerPackingReport]


@dataclass
class LayerMeta:
    name: str
    layer_type: str
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    weight_shape: Tuple[int, ...]
    out_units: int
    in_units: int
    num_params: int
    params_per_out_unit: int
    output_elements: int
    input_elements: int
    reduction_dim: int
    scalar_macs: int


class PackedMACSimulator:
    """
    Lightweight packed-MAC simulator abstraction for tile-level counting/toy checks.
    Uses separate packed operand words and 32-bit accumulation.
    """

    def __init__(self, register_size: int = 16, acc_width: int = 32):
        self.register_size = int(register_size)
        self.acc_width = int(acc_width)

    @staticmethod
    def _sign_extend(v: int, bits: int) -> int:
        if bits <= 0:
            return 0
        sign_bit = 1 << (bits - 1)
        return (v ^ sign_bit) - sign_bit

    def pack_lanes(self, lane_values: List[int], bit_width: int) -> int:
        word = 0
        mask = (1 << bit_width) - 1
        for i, v in enumerate(lane_values):
            word |= (int(v) & mask) << (i * bit_width)
        return int(word)

    def extract_lane(self, word: int, lane_idx: int, bit_width: int, signed: bool = True) -> int:
        mask = (1 << bit_width) - 1
        raw = (int(word) >> (lane_idx * bit_width)) & mask
        return self._sign_extend(raw, bit_width) if signed else int(raw)

    def simulate_tile_mac(self, w_word: int, a_word: int, lanes: int, w_bits: int, a_bits: int) -> int:
        acc = 0
        for lane in range(int(lanes)):
            w = self.extract_lane(w_word, lane, int(w_bits), signed=True)
            a = self.extract_lane(a_word, lane, int(a_bits), signed=True)
            acc += int(w) * int(a)
        # Clamp to signed accumulator range.
        lo = -(2 ** (self.acc_width - 1))
        hi = (2 ** (self.acc_width - 1)) - 1
        return int(max(lo, min(hi, acc)))


def _load_config(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "config" in data:
        return data["config"]
    return data


def _resolve_input_shape(model_name: str, dataset: str, input_shape: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if input_shape is not None:
        return input_shape
    if dataset == "fashionmnist":
        c = 1
    else:
        c = 3
    if model_name == "alexnet":
        s = 227
    elif model_name in ["levit", "swin"]:
        s = 224
    elif model_name in ["vgg11_bn", "resnet"]:
        s = 32
    else:
        s = 224
    return (c, s, s)


def _extract_layer_metadata(model: nn.Module, input_shape: Tuple[int, int, int], device: str = "cpu") -> List[LayerMeta]:
    model = model.to(device).eval()
    hook_data: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    order: List[str] = []
    hooks = []

    def _hook(name):
        def fn(module, inp, out):
            if name not in order:
                order.append(name)
            in_t = inp[0] if isinstance(inp, (tuple, list)) else inp
            hook_data[name] = {
                "in_shape": tuple(in_t.shape),
                "out_shape": tuple(out.shape),
            }
        return fn

    quant_layers: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) or (
            hasattr(module, "weight")
            and module.weight is not None
            and module.__class__.__name__.lower().startswith("q")
        ):
            quant_layers[name] = module
            hooks.append(module.register_forward_hook(_hook(name)))

    with torch.no_grad():
        x = torch.randn(1, *input_shape, device=device)
        _ = model(x)

    for h in hooks:
        h.remove()

    metas: List[LayerMeta] = []
    for name in order:
        module = quant_layers[name]
        in_shape = hook_data[name]["in_shape"]
        out_shape = hook_data[name]["out_shape"]
        w_shape = tuple(module.weight.shape) if hasattr(module, "weight") and module.weight is not None else ()
        num_params = int(module.weight.numel()) if hasattr(module, "weight") and module.weight is not None else 0

        if len(w_shape) == 4:  # conv
            out_units = int(w_shape[0])
            in_units = int(w_shape[1])
            params_per_out = int(num_params / max(out_units, 1))
            out_h, out_w = int(out_shape[-2]), int(out_shape[-1])
            output_elements = int(out_units * out_h * out_w)
            input_elements = int(in_shape[1] * in_shape[2] * in_shape[3])
            kh, kw = int(w_shape[2]), int(w_shape[3])
            groups = int(getattr(module, "groups", 1))
            reduction_dim = int((in_units // max(groups, 1)) * kh * kw)
            scalar_macs = int(output_elements * reduction_dim)
            layer_type = "conv"
        else:  # linear
            out_units = int(w_shape[0]) if len(w_shape) >= 2 else int(out_shape[-1])
            in_units = int(w_shape[1]) if len(w_shape) >= 2 else int(in_shape[-1])
            params_per_out = int(num_params / max(out_units, 1)) if num_params > 0 else int(in_units)
            output_elements = int(out_units)
            input_elements = int(in_units)
            reduction_dim = int(in_units)
            scalar_macs = int(output_elements * reduction_dim)
            layer_type = "linear"

        metas.append(
            LayerMeta(
                name=name,
                layer_type=layer_type,
                in_shape=in_shape,
                out_shape=out_shape,
                weight_shape=w_shape,
                out_units=out_units,
                in_units=in_units,
                num_params=num_params,
                params_per_out_unit=params_per_out,
                output_elements=output_elements,
                input_elements=input_elements,
                reduction_dim=reduction_dim,
                scalar_macs=scalar_macs,
            )
        )
    return metas


def _hist_from_bits(bits_cfg, total_elements: int, num_units: int) -> Dict[int, int]:
    if isinstance(bits_cfg, int):
        return {int(bits_cfg): int(total_elements)}
    if isinstance(bits_cfg, list):
        if len(bits_cfg) == 0:
            return {8: int(total_elements)}
        if len(bits_cfg) == num_units:
            elems_per_unit = int(total_elements // max(num_units, 1))
            h = defaultdict(int)
            for b in bits_cfg:
                h[int(b)] += elems_per_unit
            return dict(h)
        if len(bits_cfg) == total_elements:
            h = defaultdict(int)
            for b in bits_cfg:
                h[int(b)] += 1
            return dict(h)
        # Fallback: use proportions from list and scale to total_elements.
        cnt = Counter(int(b) for b in bits_cfg)
        h = {}
        running = 0
        keys = sorted(cnt.keys(), reverse=True)
        for i, b in enumerate(keys):
            if i == len(keys) - 1:
                h[b] = total_elements - running
            else:
                v = int(round((cnt[b] / len(bits_cfg)) * total_elements))
                h[b] = v
                running += v
        return h
    return {8: int(total_elements)}


def _channel_hist(bits_cfg, num_units: int) -> Dict[int, int]:
    if isinstance(bits_cfg, int):
        return {int(bits_cfg): int(num_units)}
    if isinstance(bits_cfg, list) and len(bits_cfg) > 0:
        if len(bits_cfg) == num_units:
            c = Counter(int(x) for x in bits_cfg)
            return dict(c)
        c = Counter(int(x) for x in bits_cfg)
        scaled = {}
        run = 0
        keys = sorted(c.keys(), reverse=True)
        for i, b in enumerate(keys):
            if i == len(keys) - 1:
                scaled[b] = num_units - run
            else:
                v = int(round((c[b] / len(bits_cfg)) * num_units))
                scaled[b] = v
                run += v
        return scaled
    return {8: int(num_units)}


def _scale_hist(hist: Dict[int, int], target_total: int) -> Dict[int, int]:
    cur_total = sum(hist.values())
    if cur_total <= 0:
        return {8: target_total}
    if cur_total == target_total:
        return dict(hist)
    out = {}
    running = 0
    keys = sorted(hist.keys(), reverse=True)
    for i, b in enumerate(keys):
        if i == len(keys) - 1:
            out[b] = max(0, target_total - running)
        else:
            v = int(round((hist[b] / cur_total) * target_total))
            out[b] = max(0, v)
            running += out[b]
    return out


def _aligned_width(b: int, aligned_policy: Dict[int, int]) -> int:
    return int(aligned_policy.get(int(b), int(b)))


def _sample_fields_from_hist(
    bit_hist: Dict[int, int],
    tensor_kind: str,
    layer_name: str,
    aligned_policy: Optional[Dict[int, int]] = None,
    limit_fields: int = 64,
) -> List[PackedField]:
    fields: List[PackedField] = []
    idx = 0
    for b in sorted(bit_hist.keys(), reverse=True):
        aligned_b = _aligned_width(b, aligned_policy) if aligned_policy else int(b)
        count = min(int(bit_hist[b]), max(0, limit_fields - idx))
        for _ in range(count):
            fields.append(
                PackedField(
                    tensor_kind=tensor_kind,
                    layer_name=layer_name,
                    group_id=f"{tensor_kind}_b{int(b)}",
                    original_index=idx,
                    bit_width=int(b),
                    aligned_bit_width=int(aligned_b),
                    register_id=-1,
                    bit_offset=-1,
                    signed=True,
                )
            )
            idx += 1
            if idx >= limit_fields:
                return fields
    return fields


def _pack_sample_registers(
    fields: List[PackedField],
    register_size: int,
    strategy: str,
    mixed_width: bool,
) -> List[PackedRegister]:
    if not fields:
        return []

    regs: List[PackedRegister] = []
    next_reg_id = 0

    def _new_reg() -> PackedRegister:
        nonlocal next_reg_id
        reg = PackedRegister(
            register_id=next_reg_id,
            register_width=register_size,
            strategy=strategy,
            used_bits=0,
            slack_bits=register_size,
            fields=[],
        )
        next_reg_id += 1
        return reg

    if not mixed_width:
        ordered = sorted(fields, key=lambda f: (f.aligned_bit_width, f.bit_width), reverse=True)
        current: Optional[PackedRegister] = None
        current_width: Optional[int] = None
        for field in ordered:
            width = int(field.aligned_bit_width)
            if current is None or current_width != width or current.used_bits + width > register_size:
                if current is not None:
                    current.slack_bits = register_size - current.used_bits
                    regs.append(current)
                current = _new_reg()
                current_width = width
            new_field = PackedField(**asdict(field))
            new_field.register_id = current.register_id
            new_field.bit_offset = current.used_bits
            current.fields.append(new_field)
            current.used_bits += width
        if current is not None:
            current.slack_bits = register_size - current.used_bits
            regs.append(current)
        return regs

    # Best-fit decreasing for mixed-width storage preview.
    ordered = sorted(fields, key=lambda f: (f.aligned_bit_width, f.bit_width), reverse=True)
    for field in ordered:
        width = int(field.aligned_bit_width)
        candidate_idx = None
        candidate_residual = None
        for i, reg in enumerate(regs):
            residual = register_size - reg.used_bits
            if residual >= width:
                post = residual - width
                if candidate_residual is None or post < candidate_residual:
                    candidate_idx = i
                    candidate_residual = post
        if candidate_idx is None:
            regs.append(_new_reg())
            candidate_idx = len(regs) - 1
        reg = regs[candidate_idx]
        new_field = PackedField(**asdict(field))
        new_field.register_id = reg.register_id
        new_field.bit_offset = reg.used_bits
        reg.fields.append(new_field)
        reg.used_bits += width
        reg.slack_bits = register_size - reg.used_bits
    return regs


def _serialize_registers(registers: List[PackedRegister]) -> List[Dict[str, object]]:
    return [asdict(r) for r in registers]


def _raw_words_and_slack(bit_hist: Dict[int, int], register_size: int, aligned_policy: Optional[Dict[int, int]] = None) -> Tuple[int, int, int]:
    words = 0
    used_bits = 0
    for b, n in bit_hist.items():
        b_eff = _aligned_width(b, aligned_policy) if aligned_policy else int(b)
        lanes = max(1, register_size // max(b_eff, 1))
        w = int(math.ceil(n / lanes))
        words += w
        used_bits += int(n * b_eff)
    allocated = int(words * register_size)
    slack = allocated - used_bits
    return words, slack, used_bits


def _heterogeneous_words_and_slack(bit_hist: Dict[int, int], register_size: int) -> Tuple[int, int, int]:
    total_bits = sum(int(b) * int(n) for b, n in bit_hist.items())
    if total_bits == 0:
        return 0, 0, 0

    bins_by_residual = defaultdict(int)
    total_bins = 0

    for b in sorted(bit_hist.keys(), reverse=True):
        n = int(bit_hist[b])
        if n <= 0:
            continue

        while n > 0:
            candidates = [r for r, c in bins_by_residual.items() if c > 0 and r >= b]
            if not candidates:
                break
            r = min(candidates)  # best fit
            c = int(bins_by_residual[r])
            q = int(r // b)
            if q <= 0:
                break
            cap = c * q
            if n >= cap:
                n -= cap
                bins_by_residual[r] -= c
                bins_by_residual[r - q * b] += c
            else:
                full_bins = n // q
                rem = n % q
                use_bins = full_bins + (1 if rem > 0 else 0)
                bins_by_residual[r] -= use_bins
                if full_bins > 0:
                    bins_by_residual[r - q * b] += full_bins
                if rem > 0:
                    bins_by_residual[r - rem * b] += 1
                n = 0
            if bins_by_residual[r] <= 0:
                bins_by_residual.pop(r, None)

        if n > 0:
            k = max(1, register_size // b)
            new_bins = int(math.ceil(n / k))
            total_bins += new_bins
            full_bins = n // k
            rem = n % k
            residual_full = register_size - k * b
            if full_bins > 0:
                bins_by_residual[residual_full] += full_bins
            if rem > 0:
                bins_by_residual[register_size - rem * b] += 1

    allocated = total_bins * register_size
    slack = allocated - total_bits
    return int(total_bins), int(slack), int(total_bits)


def _compute_tile_d(bw: int, ba: int, register_size: int, acc_width: int) -> int:
    d_w = max(1, register_size // max(bw, 1))
    d_a = max(1, register_size // max(ba, 1))
    max_prod = max(1, (2 ** max(bw - 1, 1) - 1) * (2 ** max(ba - 1, 1) - 1))
    t_acc = max(1, (2 ** max(acc_width - 1, 1) - 1) // max_prod)
    return max(1, min(d_w, d_a, t_acc))


def _compute_tile_metrics(bw: int, ba: int, register_size: int, acc_width: int) -> Tuple[int, int, int, int]:
    d_w = max(1, register_size // max(bw, 1))
    d_a = max(1, register_size // max(ba, 1))
    max_prod = max(1, (2 ** max(bw - 1, 1) - 1) * (2 ** max(ba - 1, 1) - 1))
    t_acc = max(1, (2 ** max(acc_width - 1, 1) - 1) // max_prod)
    tile = max(1, min(d_w, d_a, t_acc))
    return d_w, d_a, t_acc, tile


def _compute_packed_issues(
    meta: LayerMeta,
    weight_channel_hist: Dict[int, int],
    strategy: str,
    register_size: int,
    acc_width: int,
    aligned_policy: Dict[int, int],
) -> Tuple[int, float, int, bool]:
    total_channels = max(1, meta.out_units)
    issues = 0
    tile_weighted_sum = 0.0
    min_safe_tile = None
    overflow_risk = False
    for b, c in weight_channel_hist.items():
        frac = c / total_channels
        scalar_group = int(round(meta.scalar_macs * frac))
        if scalar_group <= 0:
            continue
        bw = int(b)
        ba = int(b)
        if strategy == "aligned":
            bw = _aligned_width(bw, aligned_policy)
            ba = _aligned_width(ba, aligned_policy)
        _, _, t_acc, d = _compute_tile_metrics(bw, ba, register_size, acc_width)
        issues += int(math.ceil(scalar_group / d))
        tile_weighted_sum += d * frac
        min_safe_tile = d if min_safe_tile is None else min(min_safe_tile, d)
        if meta.reduction_dim > t_acc:
            overflow_risk = True
    issues = max(1, int(issues))
    avg_lane_util = (meta.scalar_macs / issues) if issues > 0 else 1.0
    return issues, float(tile_weighted_sum), int(min_safe_tile or 1), bool(overflow_risk)


def _hist_to_str_count(hist: Dict[int, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(hist.items(), key=lambda kv: kv[0])}


def _make_layer_reports(
    strategy: str,
    metas: List[LayerMeta],
    weight_cfg: Dict[str, object],
    act_cfg: Dict[str, object],
    register_size: int,
    acc_width: int,
    aligned_policy: Dict[int, int],
    default_input_bits: int = 8,
) -> List[LayerPackingReport]:
    reports: List[LayerPackingReport] = []
    prev_output_hist: Optional[Dict[int, int]] = None
    prev_output_elems: Optional[int] = None
    simulator = PackedMACSimulator(register_size=register_size, acc_width=acc_width)

    for meta in metas:
        w_cfg = weight_cfg.get(meta.name, 8)
        a_cfg = act_cfg.get(meta.name, w_cfg if isinstance(w_cfg, int) else 8)

        weight_elem_hist = _hist_from_bits(w_cfg, meta.num_params, meta.out_units)
        output_elem_hist = _hist_from_bits(a_cfg, meta.output_elements, meta.out_units)
        if prev_output_hist is None:
            input_elem_hist = {default_input_bits: meta.input_elements}
        else:
            input_elem_hist = _scale_hist(prev_output_hist, meta.input_elements)

        weight_channel_hist = _channel_hist(w_cfg, meta.out_units)
        cgrp_result = None
        posthoc_result = None
        if (
            isinstance(w_cfg, list)
            and isinstance(a_cfg, list)
            and len(w_cfg) == len(a_cfg)
            and len(w_cfg) > 0
        ):
            channels = list(zip(w_cfg, a_cfg))
            cgrp_result = cgrp_pack_layer(channels, R=register_size)
            posthoc_result = posthoc_pack_layer(channels, R=register_size)

        if strategy == "raw_homogeneous":
            w_words, w_slack, w_used = _raw_words_and_slack(weight_elem_hist, register_size, None)
            a_words, a_slack, a_used = _raw_words_and_slack(input_elem_hist, register_size, None)
            o_words, o_slack, o_used = _raw_words_and_slack(output_elem_hist, register_size, None)
            packed_issues, lane_util, max_safe_tile, overflow_risk = _compute_packed_issues(
                meta, weight_channel_hist, "raw", register_size, acc_width, aligned_policy
            )
            weight_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(weight_elem_hist, "weight", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            act_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(input_elem_hist, "activation", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            out_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(output_elem_hist, "output", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            strategy_notes = "Exact-width homogeneous packing for storage and grouped compute."
        elif strategy == "aligned":
            w_words, w_slack, w_used = _raw_words_and_slack(weight_elem_hist, register_size, aligned_policy)
            a_words, a_slack, a_used = _raw_words_and_slack(input_elem_hist, register_size, aligned_policy)
            o_words, o_slack, o_used = _raw_words_and_slack(output_elem_hist, register_size, aligned_policy)
            packed_issues, lane_util, max_safe_tile, overflow_risk = _compute_packed_issues(
                meta, weight_channel_hist, "aligned", register_size, acc_width, aligned_policy
            )
            weight_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(weight_elem_hist, "weight", meta.name, aligned_policy),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            act_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(input_elem_hist, "activation", meta.name, aligned_policy),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            out_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(output_elem_hist, "output", meta.name, aligned_policy),
                    register_size,
                    strategy,
                    mixed_width=False,
                )
            )
            strategy_notes = "Aligned slot packing; 3-bit values are widened per alignment policy."
        elif strategy == "heterogeneous_storage":
            w_words, w_slack, w_used = _heterogeneous_words_and_slack(weight_elem_hist, register_size)
            a_words, a_slack, a_used = _heterogeneous_words_and_slack(input_elem_hist, register_size)
            o_words, o_slack, o_used = _heterogeneous_words_and_slack(output_elem_hist, register_size)
            packed_issues, lane_util, max_safe_tile, overflow_risk = _compute_packed_issues(
                meta, weight_channel_hist, "raw", register_size, acc_width, aligned_policy
            )
            weight_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(weight_elem_hist, "weight", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            act_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(input_elem_hist, "activation", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            out_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(output_elem_hist, "output", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            strategy_notes = "Mixed-width storage packing only; compute metrics assume regrouped exact-width execution."
        elif strategy == "hybrid_storage_compute":
            w_words, w_slack, w_used = _heterogeneous_words_and_slack(weight_elem_hist, register_size)
            a_words, a_slack, a_used = _heterogeneous_words_and_slack(input_elem_hist, register_size)
            o_words, o_slack, o_used = _heterogeneous_words_and_slack(output_elem_hist, register_size)
            packed_issues, lane_util, max_safe_tile, overflow_risk = _compute_packed_issues(
                meta, weight_channel_hist, "raw", register_size, acc_width, aligned_policy
            )
            weight_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(weight_elem_hist, "weight", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            act_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(input_elem_hist, "activation", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            out_regs_sample = _serialize_registers(
                _pack_sample_registers(
                    _sample_fields_from_hist(output_elem_hist, "output", meta.name, None),
                    register_size,
                    strategy,
                    mixed_width=True,
                )
            )
            strategy_notes = "Compact storage with regular compute-time repacking grouped by exact bit-width."
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if cgrp_result is not None:
            # OLD: packed_issues, lane_util, max_safe_tile, overflow_risk were derived from
            # OLD: the legacy histogram/simulator path alone.
            # OLD: w_words, w_slack, w_used were derived from width histograms alone.
            n_regs = cgrp_result["n_regs"]
            fill_rate = cgrp_result["fill_rate"]
            wasted_bits = cgrp_result["wasted_bits"]
            packed_issue_reduction = cgrp_result["packed_issue_reduction"]

            posthoc_n_regs = posthoc_result["n_regs"]
            posthoc_pir = posthoc_result["packed_issue_reduction"]

            w_words = n_regs
            w_used = sum(int(bw) for bw, _ in channels)
            w_slack = wasted_bits
            packed_issues = max(
                1,
                int(math.ceil(meta.scalar_macs / max(packed_issue_reduction, 1e-9))),
            )
            lane_util = fill_rate
            max_safe_tile = max(len(bin_channels) for bin_channels in cgrp_result["bins"])
            overflow_risk = False
            strategy_notes += (
                f" CGRP weight_regs={n_regs}, posthoc_weight_regs={posthoc_n_regs},"
                f" cgrp_pir={packed_issue_reduction:.4f}, posthoc_pir={posthoc_pir:.4f}."
            )

        total_regs = int(w_words + a_words + o_words)
        alloc_bits = int(total_regs * register_size)
        used_bits = int(w_used + a_used + o_used)
        slack_bits = int(w_slack + a_slack + o_slack)
        util = (used_bits / alloc_bits) if alloc_bits > 0 else 0.0
        reduction = (meta.scalar_macs / packed_issues) if packed_issues > 0 else 1.0
        requires_acc_tiling = bool(meta.reduction_dim > max_safe_tile)

        # Output-accumulator estimate for reference (upper-bound style).
        acc_regs = int(meta.output_elements)

        avg_b = 0.0
        if meta.num_params > 0:
            avg_b = sum(int(k) * int(v) for k, v in weight_elem_hist.items()) / meta.num_params

        logical_w_bits = int(sum(int(k) * int(v) for k, v in weight_elem_hist.items()))
        logical_a_bits = int(sum(int(k) * int(v) for k, v in input_elem_hist.items()))
        logical_o_bits = int(sum(int(k) * int(v) for k, v in output_elem_hist.items()))

        throughput_factor = float(reduction)
        avg_lane_utilization = float(util)

        reports.append(
            LayerPackingReport(
                layer_name=meta.name,
                layer_type=meta.layer_type,
                strategy=strategy,
                num_parameters=int(meta.num_params),
                weight_bit_distribution=_hist_to_str_count(weight_elem_hist),
                activation_bit_distribution=_hist_to_str_count(input_elem_hist),
                output_bit_distribution=_hist_to_str_count(output_elem_hist),
                average_bit_width=float(round(avg_b, 4)),
                logical_storage_bits_weight=logical_w_bits,
                logical_storage_bits_activation=logical_a_bits,
                logical_storage_bits_output=logical_o_bits,
                total_logical_storage_bits=int(logical_w_bits + logical_a_bits + logical_o_bits),
                weight_registers=int(w_words),
                activation_registers=int(a_words),
                output_registers=int(o_words),
                total_operand_registers=int(total_regs),
                accumulator_registers_estimate=int(acc_regs),
                utilization=float(round(util, 6)),
                avg_register_utilization=float(round(avg_lane_utilization, 6)),
                slack_bits=int(slack_bits),
                scalar_macs=int(meta.scalar_macs),
                packed_issues=int(packed_issues),
                reduction_factor=float(round(reduction, 6)),
                packed_issue_reduction_factor=float(round(reduction, 6)),
                lane_utilization=float(round(lane_util, 6)),
                effective_throughput_factor=float(round(throughput_factor, 6)),
                max_safe_tile_size=int(max_safe_tile),
                overflow_risk=bool(overflow_risk),
                requires_accumulator_tiling=bool(requires_acc_tiling),
                storage_words_weight=int(w_words),
                storage_words_activation=int(a_words),
                storage_words_output=int(o_words),
                example_weight_registers=weight_regs_sample,
                example_activation_registers=act_regs_sample,
                example_output_registers=out_regs_sample,
                notes=f"{strategy_notes} reduction_dim={meta.reduction_dim}, out_elems={meta.output_elements}",
            )
        )

        prev_output_hist = output_elem_hist
        prev_output_elems = meta.output_elements

    return reports


def _aggregate_global(
    strategy_name: str,
    per_layer: List[LayerPackingReport],
    baseline_registers: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> GlobalPackingReport:
    total_regs = sum(r.total_operand_registers for r in per_layer)
    total_storage = sum(r.storage_words_weight + r.storage_words_activation + r.storage_words_output for r in per_layer)
    total_logical_bits = sum(r.total_logical_storage_bits for r in per_layer)
    total_slack = sum(r.slack_bits for r in per_layer)
    scalar = sum(r.scalar_macs for r in per_layer)
    issues = sum(r.packed_issues for r in per_layer)
    reduction = (scalar / issues) if issues > 0 else 1.0
    savings = (1.0 - (total_regs / baseline_registers)) * 100 if baseline_registers > 0 else 0.0
    cost = (
        alpha * sum(r.weight_registers for r in per_layer)
        + beta * sum(r.activation_registers for r in per_layer)
        + gamma * sum(r.output_registers for r in per_layer)
        + delta * issues
    )
    return GlobalPackingReport(
        strategy_name=strategy_name,
        baseline_registers=int(baseline_registers),
        total_registers=int(total_regs),
        total_storage_words=int(total_storage),
        total_logical_storage_bits=int(total_logical_bits),
        total_slack_bits=int(total_slack),
        total_scalar_macs=int(scalar),
        total_packed_issues=int(issues),
        total_reduction_factor=float(round(reduction, 6)),
        total_effective_throughput_factor=float(round(reduction, 6)),
        savings_percent=float(round(savings, 4)),
        objective_cost=float(round(cost, 4)),
        per_layer_reports=per_layer,
    )


def _write_outputs(
    output_dir: str,
    reports: Dict[str, GlobalPackingReport],
    baseline_a: GlobalPackingReport,
    baseline_b_raw: GlobalPackingReport,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name, report in reports.items():
        path = os.path.join(output_dir, f"{name}_report.json")
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2)

    csv_path = os.path.join(output_dir, "per_layer_summary.csv")
    fieldnames = [
        "strategy",
        "layer_name",
        "layer_type",
        "num_parameters",
        "average_bit_width",
        "total_logical_storage_bits",
        "weight_registers",
        "activation_registers",
        "output_registers",
        "total_operand_registers",
        "utilization",
        "avg_register_utilization",
        "slack_bits",
        "scalar_macs",
        "packed_issues",
        "packed_issue_reduction_factor",
        "lane_utilization",
        "effective_throughput_factor",
        "max_safe_tile_size",
        "overflow_risk",
        "requires_accumulator_tiling",
        "storage_words_weight",
        "storage_words_activation",
        "storage_words_output",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sname, grec in reports.items():
            for lr in grec.per_layer_reports:
                writer.writerow(
                    {
                        "strategy": sname,
                        "layer_name": lr.layer_name,
                        "layer_type": lr.layer_type,
                        "num_parameters": lr.num_parameters,
                        "average_bit_width": lr.average_bit_width,
                        "total_logical_storage_bits": lr.total_logical_storage_bits,
                        "weight_registers": lr.weight_registers,
                        "activation_registers": lr.activation_registers,
                        "output_registers": lr.output_registers,
                        "total_operand_registers": lr.total_operand_registers,
                        "utilization": lr.utilization,
                        "avg_register_utilization": lr.avg_register_utilization,
                        "slack_bits": lr.slack_bits,
                        "scalar_macs": lr.scalar_macs,
                        "packed_issues": lr.packed_issues,
                        "packed_issue_reduction_factor": lr.packed_issue_reduction_factor,
                        "lane_utilization": lr.lane_utilization,
                        "effective_throughput_factor": lr.effective_throughput_factor,
                        "max_safe_tile_size": lr.max_safe_tile_size,
                        "overflow_risk": lr.overflow_risk,
                        "requires_accumulator_tiling": lr.requires_accumulator_tiling,
                        "storage_words_weight": lr.storage_words_weight,
                        "storage_words_activation": lr.storage_words_activation,
                        "storage_words_output": lr.storage_words_output,
                    }
                )

    md_path = os.path.join(output_dir, "global_comparison.md")
    sorted_reports = sorted(reports.values(), key=lambda r: r.objective_cost)
    with open(md_path, "w") as f:
        f.write("# Register Packing Strategy Comparison\n\n")
        f.write("## Global Table\n\n")
        f.write("| Strategy | Total Registers | Savings vs Baseline A | Total Storage Words | Logical Bits | Packed Issue Reduction | Objective Cost |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted_reports:
            f.write(
                f"| {r.strategy_name} | {r.total_registers:,} | {r.savings_percent:.2f}% | "
                f"{r.total_storage_words:,} | {r.total_logical_storage_bits:,} | {r.total_reduction_factor:.3f}x | {r.objective_cost:.2f} |\n"
            )
        f.write("\n")
        f.write(f"- Baseline A (uniform 8-bit): {baseline_a.total_registers:,} registers\n")
        f.write(f"- Baseline B (MQF raw): {baseline_b_raw.total_registers:,} registers\n")
        if "aligned" in reports:
            f.write(f"- Baseline C (MQF aligned): {reports['aligned'].total_registers:,} registers\n")
        if sorted_reports:
            f.write(f"- Best overall by objective: **{sorted_reports[0].strategy_name}**\n")

        f.write("\n## Per-Layer Best Strategy\n\n")
        layer_names = [lr.layer_name for lr in sorted_reports[0].per_layer_reports] if sorted_reports else []
        f.write("| Layer | Best Strategy | Registers |\n")
        f.write("|---|---|---:|\n")
        for lname in layer_names:
            candidates = []
            for r in sorted_reports:
                lr = next(x for x in r.per_layer_reports if x.layer_name == lname)
                candidates.append((lr.total_operand_registers, r.strategy_name))
            candidates.sort(key=lambda x: x[0])
            f.write(f"| {lname} | {candidates[0][1]} | {candidates[0][0]:,} |\n")

    txt_path = os.path.join(output_dir, "human_report.txt")
    with open(txt_path, "w") as f:
        f.write("Register Packing Optimization Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Baseline A registers: {baseline_a.total_registers:,}\n")
        f.write(f"Baseline B (MQF raw) registers: {baseline_b_raw.total_registers:,}\n")
        f.write("-" * 70 + "\n")
        for r in sorted_reports:
            f.write(
                f"{r.strategy_name:24s} | regs={r.total_registers:,} | "
                f"savings={r.savings_percent:.2f}% | storage={r.total_storage_words:,} | "
                f"packed_reduction={r.total_reduction_factor:.3f}x | cost={r.objective_cost:.2f}\n"
            )
        f.write("-" * 70 + "\n")
        if sorted_reports:
            f.write(f"Best strategy overall: {sorted_reports[0].strategy_name}\n")
            f.write("Storage vs compute separation:\n")
            for r in sorted_reports:
                f.write(
                    f"  - {r.strategy_name}: storage_words={r.total_storage_words:,}, "
                    f"packed_issue_reduction={r.total_reduction_factor:.3f}x\n"
                )
        f.write("Layers with largest register use under MQF raw:\n")
        b_raw_layers = sorted(
            baseline_b_raw.per_layer_reports,
            key=lambda x: x.total_operand_registers,
            reverse=True,
        )
        for lr in b_raw_layers[:5]:
            f.write(f"  - {lr.layer_name}: {lr.total_operand_registers:,}\n")


def run_packing_analysis(
    model: nn.Module,
    model_name: str,
    dataset: str,
    weight_config: Dict[str, object],
    activation_config: Dict[str, object],
    output_dir: str,
    register_size: int = 16,
    acc_width: int = 32,
    aligned_policy: Optional[Dict[int, int]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    default_input_bits: int = 8,
    input_shape: Optional[Tuple[int, int, int]] = None,
    device: str = "cpu",
) -> Dict[str, GlobalPackingReport]:
    aligned_policy = aligned_policy or {2: 2, 3: 4, 4: 4, 8: 8}
    in_shape = _resolve_input_shape(model_name, dataset, input_shape)
    metas = _extract_layer_metadata(model, in_shape, device=device)

    baseline_w = {m.name: 8 for m in metas}
    baseline_a = {m.name: 8 for m in metas}

    reports: Dict[str, GlobalPackingReport] = {}
    per_layer_baseline = _make_layer_reports(
        strategy="raw_homogeneous",
        metas=metas,
        weight_cfg=baseline_w,
        act_cfg=baseline_a,
        register_size=register_size,
        acc_width=acc_width,
        aligned_policy=aligned_policy,
        default_input_bits=default_input_bits,
    )
    baseline_a_report = _aggregate_global(
        "baseline_uniform_8bit",
        per_layer_baseline,
        baseline_registers=sum(r.total_operand_registers for r in per_layer_baseline),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )
    baseline_regs = baseline_a_report.total_registers

    for strategy in ["raw_homogeneous", "aligned", "heterogeneous_storage", "hybrid_storage_compute"]:
        per_layer = _make_layer_reports(
            strategy=strategy,
            metas=metas,
            weight_cfg=weight_config,
            act_cfg=activation_config,
            register_size=register_size,
            acc_width=acc_width,
            aligned_policy=aligned_policy,
            default_input_bits=default_input_bits,
        )
        reports[strategy] = _aggregate_global(
            strategy_name=strategy,
            per_layer=per_layer,
            baseline_registers=baseline_regs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

    baseline_b_raw = reports["raw_homogeneous"]
    reports["baseline_uniform_8bit"] = baseline_a_report
    _write_outputs(output_dir, reports, baseline_a_report, baseline_b_raw)
    return reports
