# Standing humanoid

Canons are on the **body**, not on a tabletop cube scene.

1. **L1** (~2.5 s VLM) — any language + camera → one JSON `{instruction, height, vx, yaw, r_arm, l_arm, done}`. No keyword FSM, no cube recipe.
2. **Tracker** (control rate, on the humanoid) — stand, COM-over-feet, squat/walk/arms from that JSON, **brace** on tilt. Does not wait for the VLM to catch a fall.
3. **Night flow** — Save fits CFM, then **H1** ablation of the error deque. Student motors only if H1 passes **and** `L2_DRIVE=1`.

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
| `L2_DRIVE` | `0` default. Student on motors only with H1 pass. |
| `STORAGE_DIR` / `PORT` | replay and HTTP |
