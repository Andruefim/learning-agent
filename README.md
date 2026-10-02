# Standing Unitree G1 (29-DoF + Dex3)

Canons are on the **body**, not on a tabletop cube scene. The live model is
`models/unitree_g1` (see `agent/h2.py` for the joint map — filename is historical).
`models/unitree_h2` is unused reference MJCF kept for comparison only.

1. **L1** (~2.5 s VLM) — any language + camera → one JSON `{instruction, height, vx, yaw, r_arm, l_arm, done}`. No keyword FSM, no cube recipe.
2. **Tracker** (control rate, on the humanoid) — stand, COM-over-feet, squat/walk/arms from that JSON, **brace** on tilt. Does not wait for the VLM to catch a fall.
3. **Oven flywheel** — Tier A privileged teacher → Tier B in-context θ → `D_success`/`D_histories` → student from scratch → DAgger → closed-loop H1. See `docs/E2E_NOTES.md` and `E2E_ZEROSHOT_FLYWHEEL.md`.
4. **Night flow** — Save fits CFM; preferred H1 is closed-loop oven ablation (`scripts/run_h1_closed_loop.py`). Interactive student residual stays at `ALPHA_MAX=0.12` unless `STUDENT_ONLY_EVAL=1` (headless only).

Default pose is **standing**. Type a command; Reset returns to the stand keyframe.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3.8:latest
python app.py
```

`python app.py --smoke` checks stand, squat, raise arm, brace (L1 cannot “save” the fall), CFM, and that drive stays off.

| Variable | Meaning |
|---|---|
| `L1_MODEL` / `L1_BASE_URL` | Ollama. WSL rewrites localhost to the Windows host. |
| `L2_DEVICE` | `cpu` / `cuda` / `rocm` |
| `E2E_ENABLE` | Deprecated π0/π0.5 whole-body teacher (`agent/e2e.py`). Default **0**. |
| `PI0_ENABLE` / `PI0_DEVICE` | Arms-only System 1.5 (off when e2e is on). |
| `STUDENT_ONLY_EVAL` | Headless oven eval: student owns 100% of the 18-D command. Interactive `ALPHA_MAX` unchanged. |
| `L2_DRIVE` | `0` default. Student on motors only with H1 pass. |
| `STORAGE_DIR` / `PORT` | replay and HTTP |

### Oven flywheel scripts

```bash
python scripts/run_episodes.py --policy zero --n 20 --gate31
python scripts/run_episodes.py --policy tier_a --n 100 --gate32a
python scripts/run_tier_b.py --all --n 100 --seed 2000 --run tier_b_gate100 --resume
python scripts/collect_flywheel_data.py --n 20 --mode k1_default
python scripts/train_student_oven.py --data data/oven_flywheel --steps 500
python scripts/run_dagger.py --rounds 3 --n 20
python scripts/run_h1_closed_loop.py --n 100
python scripts/run_flywheel_loop.py --turns 2
python scripts/validate_dataset.py --data data/oven_flywheel --min-success 300 --min-histories 200
```
