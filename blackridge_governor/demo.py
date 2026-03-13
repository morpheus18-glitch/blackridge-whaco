# Copyright (c) 2026 Blackridge Autonomy LLC. All rights reserved.
"""
Blackridge Demo Runner
=======================
WHACO (Warranted Hierarchical Autonomous Cooperative Operations)

Usage:
    python -m blackridge_governor.demo          # full demo
    python -m blackridge_governor.demo --quick   # SDK smoke test only
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run_sdk_smoke_test():
    """Verify the SDK governor produces correct outputs."""
    from blackridge_governor import THRUST_FRACS, Governor

    print("=" * 60)
    print("SDK SMOKE TEST")
    print("=" * 60)

    gov = Governor(fuel_budget=4.0)
    checks = []

    # Test 1: Basic step
    thrust, mid, mname = gov.step(
        wind_x=-3.0, wind_y=0.0, fuel_remaining=4.0,
        dist_to_goal=50.0, groundspeed=6.0, throttle_pct=0.55,
    )
    checks.append(("Step returns valid thrust", 0.0 <= thrust <= 1.0))
    checks.append(("Step returns valid mode", mname in THRUST_FRACS))

    # Test 2: Low fuel → CONSERVE
    gov.reset()
    thrust, mid, mname = gov.step(
        wind_x=0.0, wind_y=0.0, fuel_remaining=0.3,
        dist_to_goal=80.0, groundspeed=6.0, throttle_pct=0.55,
    )
    checks.append(("Low fuel → high pressure", gov.fuel_pressure > 0.5))

    # Test 3: Stall → PUNCH
    gov.reset()
    thrust, mid, mname = gov.step(
        wind_x=-8.0, wind_y=0.0, fuel_remaining=2.0,
        dist_to_goal=20.0, groundspeed=0.5, throttle_pct=0.8,
        has_wind_cov=False,
    )
    checks.append(("Stall detection → PUNCH", mname == "PUNCH"))
    checks.append(("PUNCH thrust = 1.0", thrust == 1.0))

    # Test 4: Tailwind → RIDE
    gov.reset()
    thrust, mid, mname = gov.step(
        wind_x=5.0, wind_y=0.0, fuel_remaining=4.0,
        dist_to_goal=50.0, groundspeed=10.0, throttle_pct=0.55,
    )
    checks.append(("Strong tailwind → RIDE", mname == "RIDE"))

    # Test 5: Telemetry capture
    frames = gov.telemetry()
    checks.append(("Telemetry captured", len(frames) > 0))
    checks.append(("Frame has mode_name", hasattr(frames[0], 'mode_name')))

    # Test 6: Reset
    gov.reset()
    checks.append(("Reset clears telemetry", len(gov.telemetry()) == 0))
    checks.append(("Reset returns to CRUISE", gov.mode == "CRUISE"))

    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {n_pass}/{len(checks)} checks passed")
    return n_pass == len(checks)


def main():
    parser = argparse.ArgumentParser(
        description="WHACO Governor Demo Runner — Warranted Hierarchical Autonomous Cooperative Operations",
    )
    parser.add_argument("--quick", action="store_true",
                        help="SDK smoke test only")
    parser.add_argument("--output", default="demo_output",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("WHACO (Warranted Hierarchical Autonomous Cooperative Operations)")
    print("Blackridge Governor SDK — Demo Runner v1.2.0")
    print(f"Output: {output_dir.resolve()}")
    print()

    ok = run_sdk_smoke_test()

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "version": "1.2.0",
        "results": {"sdk_smoke_test": {"passed": ok}},
    }
    report_path = output_dir / "demo_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
