#!/usr/bin/env python
"""Validate oven flywheel dataset schema (§3.3 gate helper)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/oven_flywheel")
    p.add_argument("--min-success", type=int, default=0)
    p.add_argument("--min-histories", type=int, default=0)
    args = p.parse_args()

    from agent.oven_dataset import validate_dataset_root

    report = validate_dataset_root(ROOT / args.data)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        return 1
    if report["D_success"] < args.min_success:
        print(f"D_success {report['D_success']} < {args.min_success}")
        return 1
    if report["D_histories"] < args.min_histories:
        print(f"D_histories {report['D_histories']} < {args.min_histories}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
