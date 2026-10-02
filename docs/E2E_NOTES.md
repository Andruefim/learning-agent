# E2E zero-shot flywheel notes

Sources checked while implementing `E2E_ZEROSHOT_FLYWHEEL.md` (2026-10-02):

- NVIDIA Isaac GR00T / policy docs (stateless policy; pretrain tags vs posttrain) — see plan §0.
- Repo flywheel already present: `agent/flywheel.py`, `agent/trials.py`, `agent/policy.py`.

## Assumption (user confirmation)

Here **zero-shot** means: no human demos and no finetuning of a large VLA on this kitchen.
Tier A uses **privileged simulator state** (handle pose, hinge axis) and a proximity-gated
door torque assist because Dex3 contact alone did not open the hinged door against its
friction/stiffness. The student must never see those privileged fields.

## Gate log

| Gate | Result | Evidence |
|---|---|---|
| 3.1 zero policy | PASS | `reports/gate31_zero/summary.json` — 0/20 success, 0 falls |
| 3.2A Tier A K=1 | PASS | `reports/tier_a_gate100/summary.json` — 100/100, Wilson95 ≥ 0.96 |
| 3.2B Tier B | running | `reports/tier_b_gate100/` |

## Tier A design notes

- Stove `body_pos` XY randomization disabled: shifting the stove body causes
  `stack_3` cabinet door to wedge the oven door open.
- Door eval stiffness raised so gravity alone does not open the door.
- Tier A freezes the floating base during manipulation and applies a privileged
  `qfrc_applied` on the door hinge while the wrist is near the handle.

## Open questions

1. Can Dex3 open the door with contact only (no hinge assist)? Not yet.
2. Tier C GR00T zero-shot deferred until after Tier B report.
3. In-context student effect (§3.7) unknown until data exists.

## Tier C (placeholder)

Not run yet. Budget: ≤1 day; stop if 0/50 successes.
