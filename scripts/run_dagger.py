#!/usr/bin/env python
"""DAgger rounds: student rollout → privileged teacher labels → retrain (§3.5).

β-mix starts near 0.5 and decays. Eval uses STUDENT_ONLY_EVAL semantics (student cmd).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n", type=int, default=20, help="scenes per round")
    p.add_argument("--seed", type=int, default=8000)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--beta0", type=float, default=0.5)
    p.add_argument("--ckpt", type=str, default="checkpoints/student_v0/flow_policy.pt")
    p.add_argument("--data", type=str, default="data/oven_flywheel")
    p.add_argument("--out", type=str, default="checkpoints/student_dagger")
    p.add_argument("--train-steps", type=int, default=200)
    args = p.parse_args()

    from agent.config import CHUNK, STAND_Z, VISION_STRIDE
    from agent.h2 import ACTION_DIM
    from agent.l3_cmd import clip_command
    from agent.oven_dataset import (
        EyeCamera,
        EpisodeRecorder,
        proprio_from_qpos,
        save_episode,
        teacher_cmd_18d,
    )
    from agent.oven_task import DEFAULT_THETA, OvenScene
    from agent.oven_teacher import TierATeacher
    from agent.policy import FlowPolicy, encode_instr, load_state, resolve_device

    device = resolve_device()
    policy = FlowPolicy().to(device)
    ckpt = ROOT / args.ckpt
    if ckpt.exists():
        load_state(policy, ckpt, device)
    else:
        print(f"warning: missing ckpt {ckpt}, starting from scratch", flush=True)

    data_root = ROOT / args.data
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rounds_log = []

    for rnd in range(1, int(args.rounds) + 1):
        beta = float(args.beta0) * (0.7 ** (rnd - 1))
        scene = OvenScene()
        eye = EyeCamera(scene.model)
        ok = 0
        t0 = time.perf_counter()
        for i in range(int(args.n)):
            seed = int(args.seed) + (rnd - 1) * 1000 + i
            applied = scene.reset(seed, randomize=True, theta=DEFAULT_THETA)
            teacher = TierATeacher(scene, theta=DEFAULT_THETA)
            teacher.reset(scene.theta)
            rec = EpisodeRecorder(
                prompt=str(applied.get("prompt") or scene.prompt),
                seed=seed,
                theta=dict(scene.theta),
                attempt=rnd,
            )
            language = encode_instr(rec.prompt)
            max_ticks = max(1, int(args.seconds / scene.dt))
            success = False
            fell = False
            stud_cmd = np.zeros(ACTION_DIM, dtype=np.float32)
            stud_cmd[3] = float(STAND_Z)
            for tick in range(max_ticks):
                # Observe current state, then optionally blend student arms into teacher targets
                # before the privileged teacher advances the phase machine + sim.
                img = eye.render(scene.data)
                prop = proprio_from_qpos(scene.qadr, scene.data.qpos)
                if tick % max(1, VISION_STRIDE) == 0:
                    with torch.no_grad():
                        image = torch.as_tensor(img, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                        proprio = torch.as_tensor(prop, device=device, dtype=torch.float32).unsqueeze(0)
                        lang = torch.as_tensor(language, device=device, dtype=torch.long).unsqueeze(0)
                        z = torch.zeros(1, 8, device=device, dtype=torch.float32)
                        errors = torch.zeros(1, 8, 3, device=device, dtype=torch.float32)
                        chunk = policy.sample(image, proprio, lang, z, errors, steps=1)
                        stud_cmd = clip_command(chunk[0, 0].cpu().numpy().astype(np.float32))
                use_teacher = bool(np.random.random() < float(beta))
                teacher.step()
                teach_cmd = teacher_cmd_18d(teacher._q_des)
                if not use_teacher:
                    # Student-owned arms on this tick's PD target (DAgger visits student states).
                    teacher._q_des[15:29] = stud_cmd[4:18]
                    teacher._write_ctrl()
                # Label is always the privileged teacher command for this state.
                priv = teacher._priv()
                rec.maybe_record(
                    image=img,
                    proprio=prop,
                    cmd=teach_cmd,
                    door_angle=float(priv["door_angle"]),
                    handle_pos=priv["handle_pos"],
                    hinge_pos=priv["hinge_pos"],
                )
                if scene.fell():
                    fell = True
                    break
                if scene.success_now():
                    success = True
                    break
            if success:
                ok += 1
                save_episode(
                    data_root,
                    split="D_success",
                    seed=seed,
                    attempt=100 + rnd,
                    recorder=rec,
                    meta={"mode": "dagger", "round": rnd, "beta": beta, "fell": fell},
                    success=True,
                )
            print(f"dagger r{rnd} [{i+1}/{args.n}] seed={seed} success={success} beta={beta:.3f}", flush=True)

        # Retrain briefly on growing dataset.
        train_out = out_root / f"round_{rnd}"
        rc = os.system(
            f'"{sys.executable}" -u scripts/train_student_oven.py --data {args.data} '
            f"--out {train_out.relative_to(ROOT).as_posix()} --steps {args.train_steps}"
        )
        new_ckpt = train_out / "flow_policy.pt"
        if new_ckpt.exists():
            load_state(policy, new_ckpt, device)
        rate = ok / max(args.n, 1)
        rounds_log.append(
            {
                "round": rnd,
                "beta": beta,
                "n": args.n,
                "successes": ok,
                "success_rate": rate,
                "wall_s": time.perf_counter() - t0,
                "train_exit": rc,
                "ckpt": str(new_ckpt) if new_ckpt.exists() else "",
            }
        )
        print(json.dumps(rounds_log[-1], indent=2), flush=True)

    (out_root / "dagger_summary.json").write_text(json.dumps(rounds_log, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
