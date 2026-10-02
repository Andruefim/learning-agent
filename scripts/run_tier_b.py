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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"skip bad jsonl line in {path.name}: {e}", flush=True)
    return rows


def run_mode(mode: str, n: int, seed: int, seconds: float, k_max: int, out_dir: Path, *, resume: bool = False) -> dict:
    from agent.oven_task import OvenScene
    from agent.oven_tier_b import ThetaLLM, run_scene

    # Recreate MuJoCo scene periodically — long GLFW/MuJoCo sessions hard-crash on Windows.
    SCENE_REFRESH = 10
    scene = OvenScene()
    llm = ThetaLLM() if mode == "k5_llm" else None
    path = out_dir / f"{mode}.jsonl"
    rows = _load_jsonl(path) if resume else []
    ok = sum(int(r["success"]) for r in rows)
    start_i = len(rows)
    if resume and start_i > 0:
        print(f"{mode}: resume from {start_i}/{n}", flush=True)
    t0 = time.perf_counter()
    mode_flag = "a" if resume and start_i > 0 else "w"
    with path.open(mode_flag, encoding="utf-8") as f:
        for i in range(start_i, n):
            if (i - start_i) > 0 and (i - start_i) % SCENE_REFRESH == 0:
                print(f"{mode}: refresh OvenScene at {i}/{n}", flush=True)
                scene = OvenScene()
            s = int(seed) + i
            res = run_scene(s, mode, k_max=k_max, seconds=seconds, scene=scene, llm=llm)  # type: ignore[arg-type]
            row = res.to_json()
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
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
    p.add_argument("--resume", action="store_true", help="append to existing jsonl for each mode")
    args = p.parse_args()

    run = args.run or time.strftime("tier_b_%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / run
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ("k1_default", "k5_random", "k5_llm") if args.all else (args.mode,)
    table = []
    for mode in modes:
        summary_path = out_dir / f"{mode}_summary.json"
        jsonl_path = out_dir / f"{mode}.jsonl"
        n_done = 0
        if args.resume and jsonl_path.exists():
            n_done = len(_load_jsonl(jsonl_path))
        if args.resume and n_done >= int(args.n):
            if summary_path.exists():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                # Rebuild summary from jsonl so --resume can finish a partial crash.
                rows = _load_jsonl(jsonl_path)
                ok = sum(int(r["success"]) for r in rows)
                data = {
                    "mode": mode,
                    "n": args.n,
                    "successes": ok,
                    "success_rate": ok / max(args.n, 1),
                    "wilson95": wilson(ok, args.n),
                    "mean_attempts": sum(len(r["attempts"]) for r in rows) / max(args.n, 1),
                    "total_wall_s": 0.0,
                    "path": str(jsonl_path),
                }
                summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"skip finished {mode}", flush=True)
            table.append(data)
            continue
        table.append(
            run_mode(mode, args.n, args.seed, args.seconds, args.k, out_dir, resume=args.resume)
        )

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
