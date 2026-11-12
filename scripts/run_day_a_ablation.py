#!/usr/bin/env python
"""Day A: Rule 7 + no-assist ablation stages (50 seeds). No Tier B / no rebuild.

Stages:
  s1_weld     — diagnostic weld/cheat (excluded from gates)
  s2_hook     — open fingers, arc pull (geometry sanity)
  s3_honest   — full Rule 7; base=frozen and base=free (separate rows)

Example:
  python scripts/run_day_a_ablation.py --n 50 --seed 13000 --workers 4
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


def _wilson(n_ok: int, n: int, z: float = 1.96) -> list[float]:
    n = max(int(n), 1)
    phat = n_ok / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return [max(0.0, center - half), min(1.0, center + half)]


def _one(job: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    from agent.oven_ablation import run_ablation_episode

    r = run_ablation_episode(
        int(job["seed"]),
        job["stage"],
        base=job["base"],
        seconds=float(job["seconds"]),
        scene_mod=job["scene_mod"],
    )
    return r.to_json()


def _summarize(rows: list[dict]) -> dict:
    # Valid for gate-like stats: not invalid; s1 excluded from decision metrics separately.
    n = len(rows)
    n_ok = sum(1 for r in rows if r["success"] and not r["invalid_episode"])
    n_fall = sum(1 for r in rows if r["fell"])
    n_inv = sum(1 for r in rows if r["invalid_episode"])
    forces = {
        "max_contact_force": max((r["contact"]["max_contact_force"] for r in rows), default=0.0),
        "max_contact_normal": max((r["contact"]["max_contact_normal"] for r in rows), default=0.0),
        "max_contact_friction": max((r["contact"]["max_contact_friction"] for r in rows), default=0.0),
        "max_hinge_passive_abs": max((r["contact"]["max_hinge_passive_abs"] for r in rows), default=0.0),
        "max_hinge_constraint_abs": max((r["contact"]["max_hinge_constraint_abs"] for r in rows), default=0.0),
    }
    door = rows[0]["door"] if rows else {}
    return {
        "n": n,
        "successes": n_ok,
        "success_rate": n_ok / max(n, 1),
        "wilson95": _wilson(n_ok, n),
        "falls": n_fall,
        "invalid_episode": n_inv,
        "mean_max_angle": sum(r["max_angle"] for r in rows) / max(n, 1),
        "door": door,
        "forces": forces,
    }


def _write_report(out_dir: Path, table: list[dict], contaminated: list[str]) -> Path:
    lines = [
        "# Day A report — honest oven physics ablation",
        "",
        "Rule 7 enforced in runner: no `xfrc_applied`/`qfrc_applied` on world bodies/dofs; "
        "no world qpos/qvel writes; equality/mocap only for diagnostic stage 1.",
        "",
        "Success = angle≥0.8 held ≥1s AND ≥200ms hand–handle contact while angle rising "
        "AND angle at last contact ≥0.4. Contact during hold not required.",
        "",
        "Execution path: **joint-PD teacher** (`l3_path=not_measured`). "
        "Student emits 18-D through L3 — labels from this teacher are **not** student-compatible "
        "until Day B rewires paths. `base=free` uses `joint_pd_stand_free_base` (L3 foundation "
        "not loaded in OvenScene).",
        "",
        "## Results",
        "",
        "| stage | base | scene_mod | success | Wilson 95% | falls | invalid | mean max∠ | cheat |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for t in table:
        s = t["summary"]
        lo, hi = s["wilson95"]
        lines.append(
            f"| {t['stage']} | {t['base']} | {t['scene_mod']} | "
            f"{s['success_rate']:.3f} ({s['successes']}/{s['n']}) | "
            f"[{lo:.3f}, {hi:.3f}] | {s['falls']} | {s['invalid_episode']} | "
            f"{s['mean_max_angle']:.3f} | {t.get('diagnostic_cheat', False)} |"
        )
    lines += ["", "## Door parameters (scene_mod=none)", ""]
    if table:
        d = table[0]["summary"]["door"]
        lines.append(
            f"- damping={d.get('damping')} frictionloss={d.get('frictionloss')} "
            f"stiffness={d.get('stiffness')} arm_m_approx={d.get('arm_m_approx'):.3f}"
        )
    lines += ["", "## Force probes (max over episodes in each row)", ""]
    for t in table:
        f = t["summary"]["forces"]
        lines.append(
            f"- **{t['stage']}/{t['base']}**: contact|F|={f['max_contact_force']:.3f} "
            f"(n={f['max_contact_normal']:.3f}, fr={f['max_contact_friction']:.3f}); "
            f"hinge |passive|={f['max_hinge_passive_abs']:.3f} "
            f"|constraint|={f['max_hinge_constraint_abs']:.3f}"
        )
    lines += ["", "## Contaminated (assist-era)", ""]
    for c in contaminated:
        lines.append(f"- `{c}`")
    lines += [
        "",
        "## Decision (point estimate on n=50, stage s3_honest base=free)",
        "",
    ]
    s3 = next((t for t in table if t["stage"] == "s3_honest" and t["base"] == "free"), None)
    s3f = next((t for t in table if t["stage"] == "s3_honest" and t["base"] == "frozen"), None)
    if s3:
        rate = s3["summary"]["success_rate"]
        lines.append(f"s3_honest/base=free success_rate = **{rate:.3f}**")
        if s3f:
            lines.append(
                f"s3_honest/base=frozen success_rate = **{s3f['summary']['success_rate']:.3f}** "
                f"(compare: free ≪ frozen ⇒ balance/physics is the story)"
            )
        if rate >= 0.20:
            rec = (
                "Branch ≥20%: Day B — rebuild D_success on honest teacher, rerun Tier B "
                "(default/random/LLM θ + Wilson), then BC→DAgger→student-only."
            )
        elif rate < 0.05:
            rec = (
                "Branch <5%: stop after Day A. Propose one of: "
                "(1) change action space (impedance / force hold), "
                "(2) RL skill on oven_open_angle in MJX, "
                "(3) declare blocker. **Do not choose — wait for user.**"
            )
        else:
            rec = (
                "Branch between 5–20%: raise rate via θ + Tier B on honest physics before "
                "student distillation."
            )
        lines.append("")
        lines.append(f"**Recommendation:** {rec}")
    else:
        lines.append("s3_honest/base=free missing — incomplete run.")
    lines.append("")
    path = out_dir / "day_a_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=13000)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", type=str, default="reports/day_a_ablation")
    p.add_argument("--resume", action="store_true", help="skip seeds already in stage jsonl")
    args = p.parse_args()

    from agent.oven_rules import mark_contaminated

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    contaminated = mark_contaminated(ROOT / "reports")
    # also document in docs
    notes = ROOT / "docs" / "CONTAMINATED.md"
    notes.write_text(
        "# Contaminated artifacts (assist-era)\n\n"
        "Tier A / Tier B / DAgger / student reports and checkpoints used privileged "
        "door `qfrc_applied` assist. Marked with `CONTAMINATED.md` beside each run.\n\n"
        "`reports/gate31_zero` is **not** contaminated.\n",
        encoding="utf-8",
    )

    configs = [
        ("s1_weld", "frozen", True),
        ("s2_hook", "frozen", False),
        ("s3_honest", "frozen", False),
        ("s3_honest", "free", False),
    ]
    table = []
    workers = max(1, int(args.workers))

    for stage, base, cheat in configs:
        tag = f"{stage}_{base}"
        path = out_dir / f"{tag}.jsonl"
        rows: list[dict] = []
        done_seeds: set[int] = set()
        if args.resume and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows.append(r)
                    done_seeds.add(int(r["seed"]))
                except json.JSONDecodeError:
                    pass
        jobs = [
            {
                "seed": int(args.seed) + i,
                "stage": stage,
                "base": base,
                "seconds": float(args.seconds),
                "scene_mod": "none",
            }
            for i in range(int(args.n))
            if (int(args.seed) + i) not in done_seeds
        ]
        print(
            f"=== {stage} base={base} todo={len(jobs)} done={len(done_seeds)} workers={workers} ===",
            flush=True,
        )
        t0 = time.perf_counter()
        mode = "a" if (args.resume and path.exists() and done_seeds) else "w"
        with path.open(mode, encoding="utf-8") as out_f:
            if workers == 1:
                for j, job in enumerate(jobs):
                    row = _one(job)
                    rows.append(row)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(
                        f"[{len(done_seeds)+j+1}/{args.n}] seed={row['seed']} success={row['success']} "
                        f"invalid={row['invalid_episode']} ang={row['max_angle']:.3f}",
                        flush=True,
                    )
            else:
                failed = []
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(_one, job): job["seed"] for job in jobs}
                    done = 0
                    for fut in as_completed(futs):
                        seed = futs[fut]
                        try:
                            row = fut.result()
                            rows.append(row)
                            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out_f.flush()
                            done += 1
                            print(
                                f"[{len(done_seeds)+done}/{args.n}] seed={row['seed']} success={row['success']} "
                                f"invalid={row['invalid_episode']} ang={row['max_angle']:.3f}",
                                flush=True,
                            )
                        except Exception as e:
                            failed.append(seed)
                            done += 1
                            print(f"[{len(done_seeds)+done}/{args.n}] seed={seed} FAILED {e}", flush=True)
                for seed in failed:
                    job = next(j for j in jobs if j["seed"] == seed)
                    try:
                        row = _one(job)
                        rows.append(row)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        print(f"[retry] seed={seed} success={row['success']}", flush=True)
                    except Exception as e:
                        print(f"[retry] seed={seed} FAILED {e}", flush=True)

        # unique by seed
        by_seed = {int(r["seed"]): r for r in rows}
        rows = [by_seed[s] for s in sorted(by_seed)]
        summary = _summarize(rows)
        summary["wall_s"] = time.perf_counter() - t0
        (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        table.append(
            {
                "stage": stage,
                "base": base,
                "scene_mod": "none",
                "diagnostic_cheat": cheat,
                "summary": summary,
            }
        )
        print(json.dumps({"stage": stage, "base": base, **summary}, indent=2), flush=True)

    report = _write_report(out_dir, table, contaminated)
    (out_dir / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(report.read_text(encoding="utf-8"), flush=True)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
