#!/usr/bin/env python
"""Tier B comparison: k1_default / k5_random / k5_llm → reports/<run>/

Example:
  python scripts/run_tier_b.py --mode k1_default --n 20 --seed 2000
  python scripts/run_tier_b.py --all --n 100 --seed 2000 --run tier_b_gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"


def wilson(n_ok: int, n: int, z: float = 1.96) -> list[float]:
    n = max(int(n), 1)
    phat = n_ok / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return [max(0.0, center - half), min(1.0, center + half)]


def run_mode(mode: str, n: int, seed: int, seconds: float, k_max: int, out_dir: Path) -> dict:
    from agent.oven_task import OvenScene
    from agent.oven_tier_b import ThetaLLM, run_scene

    scene = OvenScene()
    llm = ThetaLLM() if mode == "k5_llm" else None
    path = out_dir / f"{mode}.jsonl"
    ok = 0
    rows = []
    t0 = time.perf_counter()
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            s = int(seed) + i
            res = run_scene(s, mode, k_max=k_max, seconds=seconds, scene=scene, llm=llm)  # type: ignore[arg-type]
            row = res.to_json()
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ok += int(res.success)
            print(
                f"{mode} [{i+1}/{n}] seed={s} success={res.success} "
                f"attempts={len(res.attempts)} ang={res.attempts[-1].max_angle:.3f}",
                flush=True,
            )
    wall = time.perf_counter() - t0
    summary = {
        "mode": mode,
        "n": n,
        "successes": ok,
        "success_rate": ok / max(n, 1),
        "wilson95": wilson(ok, n),
        "mean_attempts": sum(len(r["attempts"]) for r in rows) / max(n, 1),
        "total_wall_s": wall,
        "path": str(path),
    }
    (out_dir / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("k1_default", "k5_random", "k5_llm"), default="k1_default")
    p.add_argument("--all", action="store_true", help="run all three modes")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--run", type=str, default="")
    args = p.parse_args()

    run = args.run or time.strftime("tier_b_%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / run
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ("k1_default", "k5_random", "k5_llm") if args.all else (args.mode,)
    table = []
    for mode in modes:
        table.append(run_mode(mode, args.n, args.seed, args.seconds, args.k, out_dir))

    report = out_dir / "tier_b_report.md"
    lines = [
        "# Tier B comparison",
        "",
        "| mode | n | success | Wilson 95% | mean attempts |",
        "|---|---:|---:|---|---:|",
    ]
    for s in table:
        lo, hi = s["wilson95"]
        lines.append(
            f"| {s['mode']} | {s['n']} | {s['success_rate']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {s['mean_attempts']:.2f} |"
        )
    lines.append("")
    lines.append(
        "Tier B proves in-context benefit only if `k5_llm` beats `k5_random` "
        "on the same seeds (non-overlapping Wilson intervals or clear gap)."
    )
    lines.append(
        "Start theta is intentionally suboptimal (`TIER_B_START_THETA`) because "
        "production Tier A defaults already saturate near 100%."
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print(report.read_text(encoding="utf-8"), flush=True)
    except UnicodeEncodeError:
        print(report.read_text(encoding="utf-8").encode("ascii", "replace").decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
