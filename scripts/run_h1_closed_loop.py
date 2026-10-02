#!/usr/bin/env python
"""Closed-loop in-context ablation for the oven student (§3.7).

Conditions on the same seeds/scenes, K attempts:
  (a) real trial context
  (b) zeroed trial context
  (c) shuffled cross-scene trial context

Writes reports/h1_closed_loop.md with Wilson intervals and McNemar tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

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


def mcnemar(a_win: int, b_win: int) -> dict:
    """Discordant pairs only. Returns chi2 (with continuity correction) and p (chi2 approx)."""
    b = int(a_win)  # a success, b fail
    c = int(b_win)  # a fail, b success
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / float(b + c)
    # Survival of chi2_1 via erfc approximation
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p": p}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="≥100 for the real gate; smaller for smoke")
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--ckpt", type=str, default="checkpoints/student_v0/flow_policy.pt")
    p.add_argument("--out", type=str, default="reports/h1_closed_loop")
    args = p.parse_args()

    os.environ["STUDENT_ONLY_EVAL"] = "1"

    from agent.config import TRIAL_MAX
    from agent.h2 import TRIAL_FEAT
    from agent.l3_cmd import clip_command
    from agent.oven_dataset import EyeCamera, proprio_from_qpos, teacher_cmd_18d
    from agent.oven_task import DEFAULT_THETA, OvenScene, oven_open_angle
    from agent.oven_teacher import TierATeacher
    from agent.h2 import STAND_Q
    from agent.policy import FlowPolicy, encode_instr, load_state, resolve_device
    import torch
    import mujoco

    device = resolve_device()
    policy = FlowPolicy().to(device)
    ckpt = ROOT / args.ckpt
    loaded = ckpt.exists() and load_state(policy, ckpt, device)
    if not loaded:
        print(f"warning: using randomly initialized student ({ckpt})", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ("real", "zero", "shuffle")
    # Per condition: list of (solved_by_k, first_success_attempt or 0)
    results = {c: [] for c in conditions}
    # Store trial tokens from a parallel bank for shuffle condition.
    token_bank: list[tuple[np.ndarray, np.ndarray]] = []

    def run_attempt(scene, eye, policy, prompt, trial_ids, trial_feat, seconds) -> dict:
        teacher = TierATeacher(scene, theta=DEFAULT_THETA)
        teacher.reset(DEFAULT_THETA)
        language = encode_instr(prompt)
        max_ticks = max(1, int(seconds / scene.dt))
        stud_cmd = np.zeros(18, dtype=np.float32)
        stud_cmd[3] = 0.78
        for tick in range(max_ticks):
            img = eye.render(scene.data)
            prop = proprio_from_qpos(scene.qadr, scene.data.qpos)
            if tick % 4 == 0:
                with torch.no_grad():
                    image = torch.as_tensor(img, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    proprio = torch.as_tensor(prop, device=device, dtype=torch.float32).unsqueeze(0)
                    lang = torch.as_tensor(language, device=device, dtype=torch.long).unsqueeze(0)
                    z = torch.zeros(1, 8, device=device, dtype=torch.float32)
                    errors = torch.zeros(1, 8, 3, device=device, dtype=torch.float32)
                    ids = torch.as_tensor(trial_ids, device=device, dtype=torch.long).unsqueeze(0)
                    feat = torch.as_tensor(trial_feat, device=device, dtype=torch.float32).unsqueeze(0)
                    chunk = policy.sample(image, proprio, lang, z, errors, skill_ids=ids, trial_feat=feat, steps=1)
                    stud_cmd = clip_command(chunk[0, 0].cpu().numpy().astype(np.float32))
            # Student-only arms; freeze base like Tier A manip.
            teacher._q_des = STAND_Q.copy()
            teacher._q_des[15:29] = stud_cmd[4:18]
            teacher._write_ctrl()
            # Hold free joint base
            scene.data.qvel[0:6] = 0.0
            mujoco.mj_step(scene.model, scene.data)
            ang = oven_open_angle(scene.model, scene.data)
            scene._angle_hist.append(ang)
            if ang >= 0.8:
                scene._success_open_s += scene.dt
            else:
                scene._success_open_s = 0.0
            if scene.fell():
                return {"success": False, "fell": True, "max_angle": float(max(scene._angle_hist or [0.0]))}
            if scene.success_now():
                return {"success": True, "fell": False, "max_angle": float(max(scene._angle_hist))}
        return {"success": False, "fell": False, "max_angle": float(max(scene._angle_hist or [0.0]))}

    def feat_from_row(success: bool, max_angle: float, fell: bool) -> tuple[np.ndarray, np.ndarray]:
        ids = np.zeros(TRIAL_MAX, dtype=np.int64)
        feat = np.zeros((TRIAL_MAX, TRIAL_FEAT), dtype=np.float32)
        # Pack a tiny summary into the first trial slot (proprio/action/error layout unused → put scalars).
        feat[0, 0] = 1.0 if success else 0.0
        feat[0, 1] = float(max_angle)
        feat[0, 2] = 1.0 if fell else 0.0
        ids[0] = 0
        return ids, feat

    scene = OvenScene()
    eye = EyeCamera(scene.model)
    n = int(args.n)
    for i in range(n):
        seed = int(args.seed) + i
        for cond in conditions:
            applied = scene.reset(seed, randomize=True, theta=DEFAULT_THETA)
            prompt = str(applied.get("prompt") or scene.prompt)
            trial_ids = np.zeros(TRIAL_MAX, dtype=np.int64)
            trial_feat = np.zeros((TRIAL_MAX, TRIAL_FEAT), dtype=np.float32)
            solved_at = 0
            for k in range(1, int(args.k) + 1):
                if cond == "zero":
                    trial_ids[:] = 0
                    trial_feat[:] = 0
                elif cond == "shuffle":
                    if token_bank:
                        ids, feat = token_bank[np.random.randint(0, len(token_bank))]
                        trial_ids, trial_feat = ids.copy(), feat.copy()
                # real: keep accumulating below
                # Re-reset same seed so scene matches across attempts.
                scene.reset(seed, randomize=True, theta=DEFAULT_THETA)
                row = run_attempt(scene, eye, policy, prompt, trial_ids, trial_feat, args.seconds)
                ids, feat = feat_from_row(row["success"], row["max_angle"], row["fell"])
                if cond == "real":
                    # shift left and append
                    trial_ids = np.roll(trial_ids, -1)
                    trial_feat = np.roll(trial_feat, -1, axis=0)
                    trial_ids[-1], trial_feat[-1] = ids[0], feat[0]
                    token_bank.append((trial_ids.copy(), trial_feat.copy()))
                if row["success"]:
                    solved_at = k
                    break
            results[cond].append(solved_at)
        print(f"scene {i+1}/{n} seed={seed} real={results['real'][-1]} zero={results['zero'][-1]} shuffle={results['shuffle'][-1]}", flush=True)
        if (i + 1) % 10 == 0:
            scene = OvenScene()
            eye = EyeCamera(scene.model)

    summary = {}
    for cond, vals in results.items():
        solved = sum(1 for v in vals if v > 0)
        by3 = sum(1 for v in vals if 0 < v <= args.k)
        first = sum(1 for v in vals if v == 1)
        gain = by3 - first
        summary[cond] = {
            "n": n,
            "solved_by_k": by3,
            "rate": by3 / max(n, 1),
            "wilson95": wilson(by3, n),
            "solved_at_1": first,
            "gain_1_to_k": gain,
            "mean_attempt_if_solved": float(np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0.0),
        }

    # McNemar real vs zero / shuffle on solved_by_k
    def discord(a: list[int], b: list[int]) -> tuple[int, int]:
        a_win = sum(1 for x, y in zip(a, b) if x > 0 and y == 0)
        b_win = sum(1 for x, y in zip(a, b) if x == 0 and y > 0)
        return a_win, b_win

    aw, zw = discord(results["real"], results["zero"])
    aw2, sw = discord(results["real"], results["shuffle"])
    tests = {
        "real_vs_zero": mcnemar(aw, zw),
        "real_vs_shuffle": mcnemar(aw2, sw),
    }
    proved = (
        summary["real"]["rate"] > summary["zero"]["rate"]
        and tests["real_vs_zero"]["p"] < 0.05
        and summary["real"]["rate"] > summary["shuffle"]["rate"]
    )
    verdict = (
        "in-context effect supported on this run"
        if proved
        else "in-context эффект студента не доказан на этом объёме данных"
    )

    blob = {"summary": summary, "mcnemar": tests, "verdict": verdict, "k": args.k, "ckpt": str(ckpt), "loaded": loaded}
    (out_dir / "summary.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")

    lines = [
        "# H1 closed-loop oven ablation",
        "",
        f"N={n} scenes, K={args.k}, seed0={args.seed}, ckpt=`{ckpt.as_posix()}` loaded={loaded}.",
        "",
        "| condition | solved by K | rate | Wilson 95% | @1 | gain 1→K |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for cond in conditions:
        s = summary[cond]
        lo, hi = s["wilson95"]
        lines.append(
            f"| {cond} | {s['solved_by_k']}/{n} | {s['rate']:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{s['solved_at_1']} | {s['gain_1_to_k']} |"
        )
    lines += [
        "",
        f"McNemar real vs zero: {tests['real_vs_zero']}",
        f"McNemar real vs shuffle: {tests['real_vs_shuffle']}",
        "",
        f"**Verdict:** {verdict}",
        "",
    ]
    report = out_dir / "h1_closed_loop.md"
    # Also publish at reports/h1_closed_loop.md as the plan requests.
    report.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "reports" / "h1_closed_loop.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
