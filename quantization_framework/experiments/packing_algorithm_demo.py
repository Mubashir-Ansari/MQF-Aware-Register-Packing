import argparse
import os
import sys
import torch

#test
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.packing import ReQAPPackingPlanner


def main():
    parser = argparse.ArgumentParser(description="REQAP packing planner demo")
    parser.add_argument("--register-size", type=int, default=16)
    parser.add_argument("--w-bits", type=int, default=2)
    parser.add_argument("--a-bits", type=int, default=2)
    parser.add_argument("--max-d", type=int, default=8)
    parser.add_argument("--n", type=int, default=64, help="vector length for packed-dot simulation")
    args = parser.parse_args()

    planner = ReQAPPackingPlanner(register_size=args.register_size, max_d=args.max_d)
    feasible = planner.feasible_factors(args.w_bits, args.a_bits)
    best_d = planner.best_factor(args.w_bits, args.a_bits)
    plan = planner.plan(args.w_bits, args.a_bits, d=best_d)

    print("=" * 70)
    print("REQAP PACKING PLAN")
    print("=" * 70)
    print(f"Register size: {args.register_size}")
    print(f"W/A bits: {args.w_bits}/{args.a_bits}")
    print(f"Feasible d: {feasible}")
    print(f"Selected d: {best_d}")
    print(f"Lane bits floor(R/d): {plan.lane_bits}")
    print(f"Eq.1 LHS/RHS: {plan.eq_lhs} < {plan.eq_rhs} => safe={plan.safe}")
    print(f"Empty bits (weight storage): {plan.empty_bits_weight}")
    print(f"Lane headroom bits: {plan.lane_headroom_bits}")
    print(f"Ops supported: {', '.join(plan.ops_supported)}")
    print("=" * 70)

    max_w = 2 ** args.w_bits - 1
    max_a = 2 ** args.a_bits - 1
    q_w = torch.randint(0, max_w + 1, (args.n,), dtype=torch.int64)
    q_a = torch.randint(0, max_a + 1, (args.n,), dtype=torch.int64)
    sim = planner.simulate_packed_dot(q_w, q_a, args.w_bits, args.a_bits, d=best_d)

    print("Packed DOT simulation:")
    print(f"  Scalar MAC ops: {int(sim['scalar_mac_ops'])}")
    print(f"  Packed MAC ops: {int(sim['packed_mac_ops'])}")
    print(f"  Ideal speedup:  {sim['ideal_speedup']:.2f}x")
    print(f"  Dot(int):       {int(sim['dot_int'])}")


if __name__ == "__main__":
    main()
