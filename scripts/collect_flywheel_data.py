#!/usr/bin/env python
"""Collect D_success / D_histories for the oven flywheel (§3.3).

Examples:
  python scripts/collect_flywheel_data.py --n 20 --seed 3000 --mode k1_default
  python scripts/collect_flywheel_data.py --n 50 --seed 3000 --mode k5_random --k 5
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="number of scenes")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=("k1_default", "k5_random", "k5_llm"), default="k1_default")
    p.add_argument("--out", type=str, default="data/oven_flywheel")
    p.add_argument("--success-only-frames", action="store_true", help="skip saving failed attempt frames to D_success")
    args = p.parse_args()

    from agent.oven_dataset import EyeCamera, save_episode, save_history_index, validate_dataset_root
    from agent.oven_task import OvenScene
    from agent.oven_tier_b import run_scene
    from agent.oven_teacher import run_tier_a_episode

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    scene = OvenScene()
    eye = EyeCamera(scene.model)
    n_ok = 0
    t0 = time.perf_counter()

    for i in range(int(args.n)):
        seed = int(args.seed) + i
        if args.mode == "k1_default":
            # Single attempt with production-ish defaults, but record frames.
            from agent.oven_task import DEFAULT_THETA

            row = run_tier_a_episode(
                seed,
                seconds=args.seconds,
                randomize=True,
                theta=DEFAULT_THETA,
                scene=scene,
                record=True,
                attempt=1,
                eye=eye,
            )
            rec = row["extras"].pop("recorder", None)
            ok = bool(row["success"])
            if rec is not None and (ok or not args.success_only_frames):
                if ok:
                    save_episode(
                        out,
                        split="D_success",
                        seed=seed,
                        attempt=1,
                        recorder=rec,
                        meta={"mode": args.mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                        success=True,
                    )
                    n_ok += 1
            print(f"[{i+1}/{args.n}] seed={seed} success={ok} frames={row['extras'].get('n_frames', 0)}", flush=True)
            continue

        # Multi-attempt: reuse Tier B loop, then re-run each attempt with recording.
        # Cheap path: call run_scene for θ schedule, then record only when needed.
        res = run_scene(seed, args.mode, k_max=args.k, seconds=args.seconds, scene=scene)
        attempts_meta = []
        saw_success = False
        for att in res.attempts:
            row = run_tier_a_episode(
                seed,
                seconds=args.seconds,
                randomize=True,
                theta=att.theta,
                scene=scene,
                record=True,
                attempt=att.k,
                eye=eye,
            )
            rec = row["extras"].pop("recorder", None)
            ok = bool(row["success"])
            attempts_meta.append(
                {
                    "k": att.k,
                    "success": ok,
                    "max_angle": float(row["max_angle"]),
                    "fell": bool(row["fell"]),
                    "contact": bool(row["extras"].get("contact")),
                    "slip": bool(row["extras"].get("slip")),
                    "min_hand_dist": float(row["extras"].get("min_hand_dist") or 1.0),
                    "outcome": row["outcome"],
                    "theta": {k: float(v) for k, v in att.theta.items()},
                }
            )
            if rec is not None:
                save_episode(
                    out,
                    split="D_histories",
                    seed=seed,
                    attempt=att.k,
                    recorder=rec,
                    meta={"mode": args.mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                    success=ok,
                )
                if ok:
                    save_episode(
                        out,
                        split="D_success",
                        seed=seed,
                        attempt=att.k,
                        recorder=rec,
                        meta={"mode": args.mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                        success=True,
                    )
                    saw_success = True
            if ok:
                break
        save_history_index(out, seed, attempts_meta)
        if saw_success:
            n_ok += 1
        print(
            f"[{i+1}/{args.n}] seed={seed} success={saw_success} attempts={len(attempts_meta)}",
            flush=True,
        )
        if (i + 1) % 10 == 0:
            scene = OvenScene()
            eye = EyeCamera(scene.model)

    report = validate_dataset_root(out)
    report["collected_success_scenes"] = n_ok
    report["wall_s"] = time.perf_counter() - t0
    (out / "collect_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
