import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PackingPlan:
    register_size: int
    w_bits: int
    a_bits: int
    d: int
    lane_bits: int
    eq_lhs: int
    eq_rhs: int
    safe: bool
    empty_bits_weight: int
    empty_bits_activation: int
    lane_headroom_bits: int
    ops_supported: Tuple[str, ...]


class ReQAPPackingPlanner:
    """
    REQAP-style register packing planner.

    Safety condition (Eq.1):
      d * (2^x - 1) * (2^y - 1) < 2^floor(R / d)
    """

    def __init__(self, register_size: int = 16, max_d: int = 8):
        self.register_size = int(register_size)
        self.max_d = int(max_d)

    def _lane_bits(self, d: int) -> int:
        if d <= 0:
            return 0
        return self.register_size // d

    def is_safe(self, w_bits: int, a_bits: int, d: int) -> bool:
        if d <= 0:
            return False
        lane_bits = self._lane_bits(d)
        if lane_bits <= 0:
            return False
        lhs = d * (2 ** int(w_bits) - 1) * (2 ** int(a_bits) - 1)
        rhs = 2 ** lane_bits
        return lhs < rhs

    def feasible_factors(self, w_bits: int, a_bits: int) -> List[int]:
        upper = min(self.max_d, self.register_size)
        feasible = []
        for d in range(1, upper + 1):
            if self.is_safe(w_bits, a_bits, d):
                feasible.append(d)
        return feasible

    def best_factor(self, w_bits: int, a_bits: int) -> int:
        feasible = self.feasible_factors(w_bits, a_bits)
        return max(feasible) if feasible else 1

    def plan(self, w_bits: int, a_bits: int, d: Optional[int] = None) -> PackingPlan:
        if d is None:
            d = self.best_factor(w_bits, a_bits)
        d = int(d)
        lane_bits = self._lane_bits(d)
        lhs = d * (2 ** int(w_bits) - 1) * (2 ** int(a_bits) - 1)
        rhs = 2 ** lane_bits if lane_bits > 0 else 0
        safe = lhs < rhs if rhs > 0 else False

        # Empty bit budget for storage packing of homogeneous values.
        empty_bits_weight = self.register_size - d * int(w_bits)
        empty_bits_activation = self.register_size - d * int(a_bits)

        # Headroom in a lane if a product is represented in that lane.
        # Positive means room remains after raw bit-growth estimate.
        lane_headroom_bits = lane_bits - (int(w_bits) + int(a_bits))

        ops = ("pack", "unpack")
        if safe:
            ops = ops + ("packed_mul", "packed_mac", "lane_accumulate")

        return PackingPlan(
            register_size=self.register_size,
            w_bits=int(w_bits),
            a_bits=int(a_bits),
            d=d,
            lane_bits=lane_bits,
            eq_lhs=lhs,
            eq_rhs=rhs,
            safe=safe,
            empty_bits_weight=empty_bits_weight,
            empty_bits_activation=empty_bits_activation,
            lane_headroom_bits=lane_headroom_bits,
            ops_supported=ops,
        )

    def pack_tensor(
        self,
        values: torch.Tensor,
        bit_width: int,
        d: Optional[int] = None,
        signed: bool = False,
    ) -> Dict[str, object]:
        """
        Packs 1-D integer tensor into register words using d lanes.
        Output words are int64 for portability.
        """
        bit_width = int(bit_width)
        if d is None:
            # Storage-only packing factor for a homogeneous integer stream.
            d = max(1, min(self.max_d, self.register_size // max(bit_width, 1)))
        d = int(d)
        if d <= 0:
            raise ValueError("d must be positive")

        v = values.detach().to(torch.int64).flatten().cpu()
        if signed:
            mask = (1 << bit_width) - 1
            v = v & mask

        words = []
        shifts = [lane * bit_width for lane in range(d)]
        mask = (1 << bit_width) - 1

        for i in range(0, v.numel(), d):
            chunk = v[i : i + d]
            word = 0
            for lane, val in enumerate(chunk.tolist()):
                word |= (int(val) & mask) << shifts[lane]
            words.append(word)

        return {
            "words": torch.tensor(words, dtype=torch.int64),
            "num_values": int(v.numel()),
            "d": d,
            "bit_width": bit_width,
            "signed": signed,
        }

    def unpack_tensor(self, packed: Dict[str, object]) -> torch.Tensor:
        words = packed["words"]
        num_values = int(packed["num_values"])
        d = int(packed["d"])
        bit_width = int(packed["bit_width"])
        signed = bool(packed.get("signed", False))

        if not isinstance(words, torch.Tensor):
            words = torch.tensor(words, dtype=torch.int64)
        words = words.to(torch.int64).flatten().cpu()

        mask = (1 << bit_width) - 1
        sign_bit = 1 << (bit_width - 1)
        out: List[int] = []

        for word in words.tolist():
            for lane in range(d):
                raw = (word >> (lane * bit_width)) & mask
                if signed and (raw & sign_bit):
                    raw = raw - (1 << bit_width)
                out.append(int(raw))
                if len(out) >= num_values:
                    break
            if len(out) >= num_values:
                break

        return torch.tensor(out, dtype=torch.int64)

    def simulate_packed_dot(
        self,
        q_w: torch.Tensor,
        q_a: torch.Tensor,
        w_bits: int,
        a_bits: int,
        d: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Simulate packed-MAC efficiency for integer vectors.
        Returns algorithmic counters, not cycle-accurate hardware timing.
        """
        if q_w.numel() != q_a.numel():
            raise ValueError("q_w and q_a must have same number of elements")

        if d is None:
            d = self.best_factor(w_bits, a_bits)
        plan = self.plan(w_bits, a_bits, d=d)

        n = int(q_w.numel())
        scalar_mac_ops = n
        packed_mac_ops = int(math.ceil(n / max(plan.d, 1)))
        speedup = scalar_mac_ops / packed_mac_ops if packed_mac_ops > 0 else 1.0

        dot_scalar = int(torch.sum(q_w.to(torch.int64) * q_a.to(torch.int64)).item())
        return {
            "safe": 1.0 if plan.safe else 0.0,
            "d": float(plan.d),
            "scalar_mac_ops": float(scalar_mac_ops),
            "packed_mac_ops": float(packed_mac_ops),
            "ideal_speedup": float(speedup),
            "dot_int": float(dot_scalar),
        }
