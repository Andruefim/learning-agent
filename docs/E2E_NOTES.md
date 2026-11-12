# E2E zero-shot flywheel notes

Sources checked while implementing `E2E_ZEROSHOT_FLYWHEEL.md` (2026-10-02):

- NVIDIA Isaac GR00T / policy docs (stateless policy; pretrain tags vs posttrain) — see plan §0.
  - https://github.com/NVIDIA/Isaac-GR00T (docs/policy.md — «the policy is stateless», checked 2026-09/10)
- Repo flywheel already present: `agent/flywheel.py`, `agent/trials.py`, `agent/policy.py`.

## Assumption (user confirmation)

Here **zero-shot** means: no human demos and no finetuning of a large VLA on this kitchen.
Tier A uses **privileged simulator state** (handle pose, hinge axis) and a proximity-gated
door torque assist because Dex3 contact alone did not open the hinged door against its
friction/stiffness. The student must never see those privileged fields.

## Gate log

| Gate | Result | Evidence |
|---|---|---|
| 3.1 zero policy | PASS (clean) | `reports/gate31_zero/summary.json` — 0/20 success, 0 falls — **not contaminated** |
| 3.2A Tier A K=1 | CONTAMINATED | privileged door `qfrc` assist — see `reports/*/CONTAMINATED.md` |
| 3.2B Tier B | CONTAMINATED | θ search over assist teacher — invalid until honest rerun |
| Day A ablation | running | `reports/day_a_ablation/` Rule 7 + stages 1/2/3 |
| 3.3 dataset | PASS | `data/oven_flywheel` — D_success=498, D_histories=217 (150 recovery), schema ok |
| 3.4 student BC | partial | val loss ↓ (1.28→0.79 BC; →0.76+trials) but open-loop MSE 0.69 > mean-baseline 0.20 — not yet Gate 3.4; ckpts `checkpoints/student_v0*` |
| 3.5 DAgger / student-only | FAIL | mixed DAgger 3×40 @100%; student-only `reports/student_only_eval100/summary.json` — **0/100** (Wilson [0,0.037]), 0 falls. Root cause: Tier A success needs privileged door torque assist; student arms-only closed loop cannot open the hinge. |
| 3.6 student-only | code ready | `STUDENT_ONLY_EVAL` in `agent/config.py` + `_mix` in `flywheel.py` |
| 3.7 H1 closed-loop | code ready | `scripts/run_h1_closed_loop.py` → `reports/h1_closed_loop.md` |
| 3.8 flywheel turns | code ready | `scripts/run_flywheel_loop.py` → `reports/flywheel.csv` |
| 3.9 cleanup | partial | `E2E_ENABLE=0` default; README G1; π0.5 path deprecated |

## Tier A design notes

- Stove `body_pos` XY randomization disabled: shifting the stove body causes
  `stack_3` cabinet door to wedge the oven door open.
- Door eval stiffness raised so gravity alone does not open the door.
- Tier A freezes the floating base during manipulation and applies a privileged
  `qfrc_applied` on the door hinge while the wrist is near the handle.

## Open questions

1. Can Dex3 open the door with contact only (no hinge assist)? Not yet.
2. Tier C GR00T zero-shot deferred until after Tier B report.
3. In-context teacher effect (§3.2B) **supported**: k5_llm 82% vs k5_random 60% on 100 shared seeds (non-overlapping Wilson 95%).
4. In-context student effect (§3.7) unknown until trained student + closed-loop H1.
5. Ego `VISION_H/W=48×64` — confirm handle visibility after first D_success dump.

## Tier C (placeholder)

Not run yet. Budget: ≤1 day; stop if 0/50 successes.
