#!/usr/bin/env python
"""Automate flywheel turns: collect → train → eval → checkpoint (§3.8).

Writes/appends reports/flywheel.csv. Decision to keep a checkpoint is success-rate
only (not MSE).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd, cwd=str(ROOT)))


def wilson(n_ok: int, n: int, z: float = 1.96) -> list[float]:
    n = max(int(n), 1)
    phat = n_ok / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return [max(0.0, center - half), min(1.0, center + half)]


def eval_student(ckpt: Path, n: int, seed: int, seconds: float) -> dict:
    """Lightweight student-only eval via Tier A scene + FlowPolicy arms (same as H1 runner core)."""
    os.environ["STUDENT_ONLY_EVAL"] = "1"
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    import mujoco
    import numpy as np
    import torch

    from agent.h2 import STAND_Q
    from agent.l3_cmd import clip_command
    from agent.oven_dataset import EyeCamera, proprio_from_qpos
    from agent.oven_task import DEFAULT_THETA, OvenScene, oven_open_angle
    from agent.policy import FlowPolicy, encode_instr, load_state, resolve_device

    device = resolve_device()
    policy = FlowPolicy().to(device)
    loaded = ckpt.exists() and load_state(policy, ckpt, device)
    from agent.joint_pd import compute_torques

    scene = OvenScene()
    eye = EyeCamera(scene.model)
    ok = falls = 0
    t_run = time.time()
    for i in range(n):
        s = seed + i
        applied = scene.reset(s, randomize=True, theta=DEFAULT_THETA)
        prompt = str(applied.get("prompt") or scene.prompt)
        language = encode_instr(prompt)
        stud_cmd = np.zeros(18, dtype=np.float32)
        stud_cmd[3] = 0.78
        max_ticks = max(1, int(seconds / scene.dt))
        success = fell = False
        t_ep = time.time()
        for tick in range(max_ticks):
            # Render only when the policy queries vision (VISION_STRIDE=4).
            if tick % 4 == 0:
                img = eye.render(scene.data)
                prop = proprio_from_qpos(scene.qadr, scene.data.qpos)
                with torch.no_grad():
                    image = torch.as_tensor(img, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    proprio = torch.as_tensor(prop, device=device, dtype=torch.float32).unsqueeze(0)
                    lang = torch.as_tensor(language, device=device, dtype=torch.long).unsqueeze(0)
                    z = torch.zeros(1, 8, device=device, dtype=torch.float32)
                    errors = torch.zeros(1, 8, 3, device=device, dtype=torch.float32)
                    chunk = policy.sample(image, proprio, lang, z, errors, steps=1)
                    stud_cmd = clip_command(chunk[0, 0].cpu().numpy().astype(np.float32))
            q = STAND_Q.copy()
            q[15:29] = stud_cmd[4:18]
            tau = compute_torques(scene.model, scene.data, q, scene.kp, scene.kd, scene.qadr, scene.vadr)
            scene.data.ctrl[:29] = tau
            scene.data.qvel[0:6] = 0.0
            mujoco.mj_step(scene.model, scene.data)
            ang = oven_open_angle(scene.model, scene.data)
            scene._angle_hist.append(ang)
            if ang >= 0.8:
                scene._success_open_s += scene.dt
            else:
                scene._success_open_s = 0.0
            if scene.fell():
                fell = True
                break
            if scene.success_now():
                success = True
                break
        ok += int(success)
        falls += int(fell)
        print(
            f"eval [{i+1}/{n}] seed={s} success={success} fell={fell} "
            f"rate={ok/max(i+1,1):.3f} wall={time.time()-t_ep:.1f}s",
            flush=True,
        )
        if (i + 1) % 10 == 0:
            try:
                eye.close()
            except Exception:
                pass
            scene = OvenScene()
            eye = EyeCamera(scene.model)
    return {
        "n": n,
        "successes": ok,
        "success_rate": ok / max(n, 1),
        "falls": falls,
        "wilson95": wilson(ok, n),
        "loaded": loaded,
        "ckpt": str(ckpt),
        "wall_s": time.time() - t_run,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--collect-n", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--data", type=str, default="data/oven_flywheel")
    p.add_argument("--seconds", type=float, default=12.0)
    args = p.parse_args()

    py = sys.executable
    csv_path = ROOT / "reports" / "flywheel.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    prev_rate = -1.0

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "version",
                "turn",
                "data_success",
                "data_histories",
                "success_rate",
                "wilson_lo",
                "wilson_hi",
                "falls",
                "keep",
                "note",
                "wall_s",
            ],
        )
        if write_header:
            w.writeheader()

        for turn in range(1, int(args.turns) + 1):
            t0 = time.time()
            ver = f"student_v{turn}"
            ckpt_dir = ROOT / "checkpoints" / ver
            # Collect more teacher successes each turn.
            run(
                [
                    py,
                    "-u",
                    "scripts/collect_flywheel_data.py",
                    "--n",
                    str(args.collect_n),
                    "--seed",
                    str(args.seed + turn * 100),
                    "--mode",
                    "k1_default",
                    "--out",
                    args.data,
                    "--seconds",
                    str(args.seconds),
                ]
            )
            run(
                [
                    py,
                    "-u",
                    "scripts/train_student_oven.py",
                    "--data",
                    args.data,
                    "--out",
                    f"checkpoints/{ver}",
                    "--steps",
                    str(args.train_steps),
                ]
            )
            ckpt = ckpt_dir / "flow_policy.pt"
            ev = eval_student(ckpt, args.eval_n, seed=args.seed + 5000 + turn * 100, seconds=args.seconds)
            from agent.oven_dataset import validate_dataset_root

            ds = validate_dataset_root(ROOT / args.data)
            keep = ev["success_rate"] >= prev_rate - 1e-9
            note = "keep (non-decreasing success)" if keep else "reject (success dropped)"
            if ev["success_rate"] > prev_rate:
                note = "keep (success improved)"
            if keep:
                prev_rate = ev["success_rate"]
            lo, hi = ev["wilson95"]
            row = {
                "version": ver,
                "turn": turn,
                "data_success": ds.get("D_success", 0),
                "data_histories": ds.get("D_histories", 0),
                "success_rate": f"{ev['success_rate']:.4f}",
                "wilson_lo": f"{lo:.4f}",
                "wilson_hi": f"{hi:.4f}",
                "falls": ev["falls"],
                "keep": int(keep),
                "note": note,
                "wall_s": f"{time.time() - t0:.1f}",
            }
            w.writerow(row)
            f.flush()
            (ckpt_dir / "eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
            print(json.dumps(row, indent=2), flush=True)

    print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
