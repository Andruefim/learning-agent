#!/usr/bin/env python
"""Headless oven episodes → reports/<run>/episodes.jsonl

Examples:
  python scripts/run_episodes.py --policy zero --n 20 --seconds 5 --run gate31_zero
  python scripts/run_episodes.py --policy zero --n 20 --workers 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else ("glfw" if os.name == "nt" else "egl")


def _one(args: tuple) -> dict:
    seed, seconds, randomize, policy = args
    from agent.oven_task import run_zero_policy_episode
    from agent.oven_teacher import run_tier_a_episode

    if policy == "zero":
        return run_zero_policy_episode(seed, seconds=seconds, randomize=randomize).to_json()
    if policy == "tier_a":
        return run_tier_a_episode(seed, seconds=seconds, randomize=randomize)
    raise ValueError(f"unsupported policy={policy!r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Headless oven episode runner")
    p.add_argument("--n", type=int, default=20, help="number of episodes")
    p.add_argument("--seed", type=int, default=0, help="base seed (episode i uses seed+i)")
    p.add_argument("--seconds", type=float, default=60.0, help="sim time limit per episode")
    p.add_argument("--policy", choices=("zero", "tier_a"), default="zero")
    p.add_argument("--workers", type=int, default=1, help="process pool size")
    p.add_argument("--run", type=str, default="", help="reports/<run> name")
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--gate31", action="store_true", help="assert 0 success and 0 falls")
    args = p.parse_args()

    run = args.run or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"

    jobs = [
        (int(args.seed) + i, float(args.seconds), not args.no_randomize, args.policy)
        for i in range(int(args.n))
    ]
    results: list[dict] = []
    t0 = time.perf_counter()
    if int(args.workers) <= 1:
        from agent.oven_task import OvenScene, run_zero_policy_episode
        from agent.oven_teacher import run_tier_a_episode

        scene = OvenScene()
        for seed, seconds, randomize, policy in jobs:
            if policy == "zero":
                results.append(
                    run_zero_policy_episode(seed, seconds=seconds, randomize=randomize, scene=scene).to_json()
                )
            elif policy == "tier_a":
                results.append(run_tier_a_episode(seed, seconds=seconds, randomize=randomize, scene=scene))
            else:
                raise ValueError(policy)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: int(r["seed"]))

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in results if r["success"])
    n_fall = sum(1 for r in results if r["fell"])
    wall = time.perf_counter() - t0
    mean_wall = float(np_mean([r["wall_s"] for r in results]))
    mean_sim = float(np_mean([r["sim_s"] for r in results]))
    summary = {
        "run": run,
        "n": len(results),
        "policy": args.policy,
        "successes": n_ok,
        "falls": n_fall,
        "mean_wall_s": mean_wall,
        "mean_sim_s": mean_sim,
        "total_wall_s": wall,
        "episodes_per_hour": (len(results) / wall) * 3600.0 if wall > 0 else 0.0,
        "path": str(out_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.gate31:
        if n_ok != 0 or n_fall != 0:
            print(f"GATE 3.1 FAILED: successes={n_ok} falls={n_fall}", flush=True)
            return 1
        print("GATE 3.1 PASSED: 0 successes, 0 falls", flush=True)
    return 0


def np_mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


if __name__ == "__main__":
    raise SystemExit(main())
