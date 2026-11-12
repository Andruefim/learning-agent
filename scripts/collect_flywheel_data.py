#!/usr/bin/env python
"""Collect D_success / D_histories for the oven flywheel (§3.3).

Speed notes:
  - single-pass recording (no re-run of attempts)
  - --workers N process pool (each worker owns MuJoCo + GL)
  - --record-stride to thin ego frames (default 8; config VISION_STRIDE is 4)
  - --resume skips seeds that already have history.json / D_success meta

Examples:
  python scripts/collect_flywheel_data.py --n 150 --mode k1_default --workers 6
  python scripts/collect_flywheel_data.py --n 100 --mode k5_random --workers 4 --resume
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
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"


def _seed_done(out: Path, mode: str, seed: int) -> bool:
    if mode == "k1_default":
        return (out / "D_success" / f"seed_{int(seed):06d}" / "attempt_01" / "meta.json").exists()
    return (out / "D_histories" / f"seed_{int(seed):06d}" / "history.json").exists()


def _collect_one(job: dict) -> dict:
    """Worker: one scene, own MuJoCo context. Returns a small status dict."""
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    from agent.oven_dataset import EyeCamera, save_episode, save_history_index
    from agent.oven_task import DEFAULT_THETA, OvenScene
    from agent.oven_tier_b import TIER_B_START_THETA, random_theta
    from agent.oven_teacher import run_tier_a_episode
    from agent.oven_task import clamp_theta
    import numpy as np

    seed = int(job["seed"])
    mode = str(job["mode"])
    out = Path(job["out"])
    seconds = float(job["seconds"])
    k_max = int(job["k"])
    stride = int(job["record_stride"])
    save_failures = bool(job["save_failures"])

    scene = OvenScene()
    eye = EyeCamera(scene.model)
    t0 = time.perf_counter()

    if mode == "k1_default":
        row = run_tier_a_episode(
            seed,
            seconds=seconds,
            randomize=True,
            theta=DEFAULT_THETA,
            scene=scene,
            record=True,
            attempt=1,
            eye=eye,
            record_stride=stride,
        )
        rec = row["extras"].pop("recorder", None)
        ok = bool(row["success"])
        if ok and rec is not None:
            save_episode(
                out,
                split="D_success",
                seed=seed,
                attempt=1,
                recorder=rec,
                meta={"mode": mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                success=True,
            )
        return {
            "seed": seed,
            "mode": mode,
            "success": ok,
            "attempts": 1,
            "frames": int(row["extras"].get("n_frames") or 0),
            "wall_s": time.perf_counter() - t0,
        }

    # Multi-attempt single pass (record while trying — no second replay).
    rng = np.random.default_rng(seed + 17)
    theta = clamp_theta(TIER_B_START_THETA)
    attempts_meta = []
    saw_success = False
    total_frames = 0
    for k in range(1, k_max + 1):
        if k > 1:
            if mode == "k5_random":
                theta = random_theta(rng, base=TIER_B_START_THETA)
            else:
                # k5_llm in workers: local ThetaLLM (may talk to Ollama).
                from agent.oven_tier_b import ThetaLLM, AttemptRecord

                hist = [
                    AttemptRecord(
                        k=a["k"],
                        theta=a["theta"],
                        max_angle=a["max_angle"],
                        success=a["success"],
                        fell=a["fell"],
                        contact=a["contact"],
                        slip=a["slip"],
                        min_hand_dist=a["min_hand_dist"],
                        outcome=a["outcome"],
                        reason=a.get("reason", ""),
                    )
                    for a in attempts_meta
                ]
                llm = ThetaLLM()
                theta, _ = llm.propose(hist, "open the oven")
        row = run_tier_a_episode(
            seed,
            seconds=seconds,
            randomize=True,
            theta=theta,
            scene=scene,
            record=True,
            attempt=k,
            eye=eye,
            record_stride=stride,
        )
        rec = row["extras"].pop("recorder", None)
        ok = bool(row["success"])
        total_frames += int(row["extras"].get("n_frames") or 0)
        attempts_meta.append(
            {
                "k": k,
                "success": ok,
                "max_angle": float(row["max_angle"]),
                "fell": bool(row["fell"]),
                "contact": bool(row["extras"].get("contact")),
                "slip": bool(row["extras"].get("slip")),
                "min_hand_dist": float(row["extras"].get("min_hand_dist") or 1.0),
                "outcome": row["outcome"],
                "reason": row.get("reason", ""),
                "theta": {kk: float(vv) for kk, vv in theta.items()},
            }
        )
        if rec is not None and (ok or save_failures):
            save_episode(
                out,
                split="D_histories",
                seed=seed,
                attempt=k,
                recorder=rec,
                meta={"mode": mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                success=ok,
            )
            if ok:
                save_episode(
                    out,
                    split="D_success",
                    seed=seed,
                    attempt=k,
                    recorder=rec,
                    meta={"mode": mode, "outcome": row["outcome"], "max_angle": row["max_angle"]},
                    success=True,
                )
                saw_success = True
        if ok:
            break
    save_history_index(out, seed, attempts_meta)
    return {
        "seed": seed,
        "mode": mode,
        "success": saw_success,
        "attempts": len(attempts_meta),
        "frames": total_frames,
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="number of scenes")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=("k1_default", "k5_random", "k5_llm"), default="k1_default")
    p.add_argument("--out", type=str, default="data/oven_flywheel")
    p.add_argument("--workers", type=int, default=1, help="parallel MuJoCo processes")
    p.add_argument("--record-stride", type=int, default=8, help="ego sample every N physics ticks")
    p.add_argument("--resume", action="store_true", help="skip seeds already on disk")
    p.add_argument(
        "--save-failures",
        action="store_true",
        default=True,
        help="keep failed attempt frames in D_histories (default on)",
    )
    p.add_argument("--no-save-failures", action="store_true", help="only write successful attempt frames")
    args = p.parse_args()

    from agent.oven_dataset import validate_dataset_root

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_failures = not bool(args.no_save_failures)

    jobs = []
    skipped = 0
    for i in range(int(args.n)):
        seed = int(args.seed) + i
        if args.resume and _seed_done(out, args.mode, seed):
            skipped += 1
            continue
        jobs.append(
            {
                "seed": seed,
                "mode": args.mode,
                "out": str(out),
                "seconds": float(args.seconds),
                "k": int(args.k),
                "record_stride": int(args.record_stride),
                "save_failures": save_failures,
            }
        )
    print(
        f"collect mode={args.mode} jobs={len(jobs)} skipped={skipped} "
        f"workers={args.workers} stride={args.record_stride}",
        flush=True,
    )
    if not jobs:
        report = validate_dataset_root(out)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report.get("ok") else 1

    n_ok = 0
    t0 = time.perf_counter()
    done = 0
    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            row = _collect_one(job)
            done += 1
            n_ok += int(row["success"])
            print(
                f"[{done}/{len(jobs)}] seed={row['seed']} success={row['success']} "
                f"attempts={row['attempts']} frames={row['frames']} wall={row['wall_s']:.1f}s",
                flush=True,
            )
    else:
        # Windows spawn: guard with __main__ already; each worker imports fresh.
        # Do not abort the whole run if one worker OOMs — record failures and continue.
        failed_seeds: list[int] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_collect_one, job): job["seed"] for job in jobs}
            for fut in as_completed(futs):
                seed = int(futs[fut])
                try:
                    row = fut.result()
                except Exception as e:
                    failed_seeds.append(seed)
                    done += 1
                    print(f"[{done}/{len(jobs)}] seed={seed} FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                done += 1
                n_ok += int(row["success"])
                print(
                    f"[{done}/{len(jobs)}] seed={row['seed']} success={row['success']} "
                    f"attempts={row['attempts']} frames={row['frames']} wall={row['wall_s']:.1f}s",
                    flush=True,
                )
        # Retry OOM/failures serially (low peak memory).
        if failed_seeds:
            print(f"retry serial for {len(failed_seeds)} failed seeds...", flush=True)
            by_seed = {int(j["seed"]): j for j in jobs}
            for seed in failed_seeds:
                job = by_seed[seed]
                try:
                    row = _collect_one(job)
                    n_ok += int(row["success"])
                    print(
                        f"[retry] seed={row['seed']} success={row['success']} "
                        f"attempts={row['attempts']} wall={row['wall_s']:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[retry] seed={seed} FAILED again: {type(e).__name__}: {e}", flush=True)

    report = validate_dataset_root(out)
    report["collected_success_scenes"] = n_ok
    report["skipped_existing"] = skipped
    report["wall_s"] = time.perf_counter() - t0
    report["workers"] = workers
    (out / "collect_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    # Windows ProcessPoolExecutor needs this.
    raise SystemExit(main())
