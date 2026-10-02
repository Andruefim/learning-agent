#!/usr/bin/env python
"""Train FlowPolicy from scratch on oven D_success / D_histories (§3.4).

- Default: mask trial tokens (pure BC).
- --with-trials: attend to trial_feat from D_histories when present.
Validation split is by scene seed (10%), not by tick.
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


def _list_episodes(data_root: Path, *, prefer_histories: bool) -> list[Path]:
    paths: list[Path] = []
    if prefer_histories:
        root = data_root / "D_histories"
        if root.exists():
            paths.extend(p.parent for p in root.rglob("student.npz"))
    root = data_root / "D_success"
    if root.exists():
        paths.extend(p.parent for p in root.rglob("student.npz"))
    # unique preserve order
    seen = set()
    out = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _seed_of(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed_"):
            return int(part.split("_", 1)[1])
    return 0


def _load_windows(ep_dir: Path, *, with_trials: bool, chunk: int):
    from agent.oven_dataset import chunkify_cmd

    with np.load(ep_dir / "student.npz") as z:
        images = z["images"]
        proprio = z["proprio"]
        cmd = z["cmd"]
        language = z["language"]
        zvec = z["z"]
        errors = z["errors"]
        trial_ids = z["trial_ids"]
        trial_feat = z["trial_feat"]
    if images.shape[0] == 0:
        return []
    chunks = chunkify_cmd(cmd, chunk=chunk)
    # Align frames to chunk starts.
    n = chunks.shape[0]
    rows = []
    for i in range(n):
        t0 = i * chunk
        if t0 >= images.shape[0]:
            break
        img = images[t0]
        pr = proprio[t0]
        rows.append(
            {
                "image": img,
                "proprio": pr,
                "language": language,
                "z": zvec,
                "errors": errors,
                "chunk": chunks[i],
                "trial_ids": trial_ids if with_trials else np.zeros_like(trial_ids),
                "trial_feat": trial_feat if with_trials else np.zeros_like(trial_feat),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/oven_flywheel")
    p.add_argument("--out", type=str, default="checkpoints/student_v0")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--with-trials", action="store_true")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    from agent.config import CHUNK, INSTR_BYTES, TRIAL_MAX
    from agent.h2 import ACTION_DIM, N_ACT, TRIAL_FEAT
    from agent.oven_dataset import validate_dataset_root
    from agent.policy import FlowPolicy, resolve_device

    data_root = ROOT / args.data
    report = validate_dataset_root(data_root)
    if report["D_success"] == 0 and report["D_histories"] == 0:
        print("no episodes found — run scripts/collect_flywheel_data.py first", flush=True)
        return 1

    eps = _list_episodes(data_root, prefer_histories=bool(args.with_trials))
    seeds = sorted({_seed_of(e) for e in eps})
    rng = np.random.default_rng(0)
    rng.shuffle(seeds)
    n_val = max(1, int(round(0.10 * len(seeds)))) if len(seeds) >= 10 else max(1, len(seeds) // 5 or 1)
    val_seeds = set(seeds[:n_val])
    train_eps = [e for e in eps if _seed_of(e) not in val_seeds]
    val_eps = [e for e in eps if _seed_of(e) in val_seeds]
    if not train_eps:
        train_eps, val_eps = eps, eps[:1]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for e in train_eps:
        train_rows.extend(_load_windows(e, with_trials=args.with_trials, chunk=CHUNK))
    for e in val_eps:
        val_rows.extend(_load_windows(e, with_trials=args.with_trials, chunk=CHUNK))
    if not train_rows:
        print("no training windows (empty frames)", flush=True)
        return 1

    device = resolve_device(args.device or None)
    policy = FlowPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(args.lr))

    def batchify(rows: list[dict], idx: np.ndarray):
        b = [rows[int(i)] for i in idx]
        image = torch.as_tensor(np.stack([x["image"] for x in b]), device=device, dtype=torch.float32)
        image = image.permute(0, 3, 1, 2) / 255.0
        proprio = torch.as_tensor(np.stack([x["proprio"] for x in b]), device=device, dtype=torch.float32)
        language = torch.as_tensor(np.stack([x["language"] for x in b]), device=device, dtype=torch.long)
        z = torch.as_tensor(np.stack([x["z"] for x in b]), device=device, dtype=torch.float32)
        errors = torch.as_tensor(np.stack([x["errors"] for x in b]), device=device, dtype=torch.float32)
        chunk = torch.as_tensor(np.stack([x["chunk"] for x in b]), device=device, dtype=torch.float32)
        trial_ids = torch.as_tensor(np.stack([x["trial_ids"] for x in b]), device=device, dtype=torch.long)
        trial_feat = torch.as_tensor(np.stack([x["trial_feat"] for x in b]), device=device, dtype=torch.float32)
        return image, proprio, language, z, errors, chunk, trial_ids, trial_feat

    def eval_loss(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        policy.eval()
        idx = np.arange(min(len(rows), max(args.batch * 4, args.batch)))
        with torch.no_grad():
            tensors = batchify(rows, idx)
            loss = policy.cfm_loss(*tensors[:6], skill_ids=tensors[6], trial_feat=tensors[7])
        return float(loss.item())

    # Mean-chunk baseline open-loop MSE on val.
    mean_chunk = np.mean(np.stack([r["chunk"] for r in train_rows]), axis=0)
    if val_rows:
        baseline = float(np.mean([(r["chunk"] - mean_chunk) ** 2 for r in val_rows]))
    else:
        baseline = float("nan")

    history = []
    t0 = time.perf_counter()
    policy.train()
    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train_rows), size=min(args.batch, len(train_rows)))
        tensors = batchify(train_rows, idx)
        loss = policy.cfm_loss(*tensors[:6], skill_ids=tensors[6], trial_feat=tensors[7])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            vloss = eval_loss(val_rows)
            history.append({"step": step, "train": float(loss.item()), "val": vloss})
            print(f"step {step}/{args.steps} train={loss.item():.4f} val={vloss:.4f}", flush=True)

    # Open-loop student vs mean baseline on val (first action of chunk).
    policy.eval()
    if val_rows:
        idx = np.arange(min(len(val_rows), 64))
        tensors = batchify(val_rows, idx)
        with torch.no_grad():
            pred = policy.sample(*tensors[:5], skill_ids=tensors[6], trial_feat=tensors[7], steps=2)
        target = tensors[5]
        ol = float(torch.mean((pred - target) ** 2).item())
    else:
        ol = float("nan")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "flow_policy.pt"
    torch.save(policy.state_dict(), ckpt)
    summary = {
        "checkpoint": str(ckpt),
        "steps": args.steps,
        "with_trials": bool(args.with_trials),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_windows": len(train_rows),
        "n_val_windows": len(val_rows),
        "val_seeds": sorted(val_seeds),
        "history": history,
        "open_loop_mse": ol,
        "mean_chunk_baseline_mse": baseline,
        "beats_mean_baseline": bool(ol < baseline) if np.isfinite(ol) and np.isfinite(baseline) else False,
        "wall_s": time.perf_counter() - t0,
        "shapes": {
            "action_dim": ACTION_DIM,
            "n_act": N_ACT,
            "chunk": CHUNK,
            "instr_bytes": INSTR_BYTES,
            "trial_max": TRIAL_MAX,
            "trial_feat": TRIAL_FEAT,
        },
        "dataset": report,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
