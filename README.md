# Standing Unitree G1 (29-DoF + Dex3) — SLEEP Continual Learning

Free-standing Unitree G1 humanoid in MuJoCo (`models/unitree_g1/scene_app.xml`) driven by the **SLEEP** Wake / NREM / REM continual learning architecture (`arXiv:2606.03979`), coupled with a closed-loop RL locomotion policy (`G1LegPolicy`) and dual Cloud Gemini / Local Ollama VLM grounding.

## Architecture

1. **System 0 (`agent/system0.py`) — 50 Hz Locomotion & Whole-Body Control**:
   - Free-standing RL gait (`models/unitree_g1/motion.pt`) + CoM inverted-pendulum balance compensation.
   - 6-DoF Damped Least Squares (DLS) whole-body Jacobian IK over waist + 7-DoF arm and Dex3-1 C-hook hand poses (`agent/dex3.py`).
   - 20-D action space (`vx, wz, squat, waist_y, waist_p, r_arm(7), l_arm(7), r_grip`) & 28-D proprioception vector.
2. **System 1 (`agent/sleep/`) — SLEEP Generalist VLA + CMS**:
   - ResNet-18 vision encoder + RU/EN subword tokenizer + Transformer + 20-D autoregressive Gaussian head + Sparse Contextual Memory Slots (`CMSChain`, `MoEBlock`).
   - **Confidence Gate**: executes System 1 closed-loop when predictive uncertainty $\sigma \le \tau_{\text{think}}$; escalates to System 2 when $\sigma > \tau_{\text{think}}$.
   - **Wake / Micro-Sleep / Deep Sleep**: online Eq. 2 updates during Wake; NREM consolidation (GKD + LTI REINFORCE + slot pruning) and REM Dreaming (random-expert hallucinated instructions, LESS cosine-similarity gradient retrieval, isolated SFT, ReST-EM) during idle Micro-Sleep or `STAND-BY (СОН)`. Checkpoints saved to `checkpoints/sleep_vla.pt`.
3. **System 2 (`agent/system2.py`) & VLM (`agent/planner.py`)**:
   - Primary **Cloud Gemini** (`GEMINI_API_KEY`, `GEMINI_MODEL`) with automatic fallback to local **Ollama** (`L1_BASE_URL`, `L1_MODEL`) on HTTP/network/timeout/parse errors (e.g. `503`).
   - MuJoCo `mj_ray` 3D visual grounding (`uv_to_world`), closed-loop walking + 6D IK primitives, and visual before/after `judge` verification.
4. **HAL Dashboard (`app.py`, `static/index.html`)**:
   - `POWER ON/OFF`, `STAND-BY (СОН)`, `τ_think` slider, live `σ` vs `τ_think` chart, VLM active backend indicator, CMS slot counter, quick Russian/English task chips, and mouse drag-orbit camera.

## Setup & Run

> **Note (Windows / AMD ROCm):** If `.venv` only has CPU PyTorch without `torchvision`, run with the system Python environment that has PyTorch + ROCm + `torchvision` + `mujoco` installed (`python app.py` / `C:\Python312\python.exe`).

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY (and ensure Ollama has qwen3.5:9b pulled for offline/503 fallback)
python app.py
```

Open `http://127.0.0.1:8000` in your browser.

### Verification

```bash
python tests/smoke.py
# or
python app.py --smoke
```

## Environment Variables

| Variable | Default / Meaning |
|---|---|
| `GEMINI_MODEL` / `GEMINI_API_KEY` | Primary Cloud Gemini VLM (`gemini-3.5-flash-lite`) |
| `L1_MODEL` / `L1_BASE_URL` | Fallback local Ollama VLM (`qwen3.5:9b` at `http://127.0.0.1:11434`) |
| `SLEEP_DEVICE` | Compute device for SLEEP VLA (`cuda` / `cpu`) |
| `MUJOCO_GL` | Offscreen OpenGL backend (`glfw` on Windows/macOS, `egl` on Linux headless) |
| `STORAGE_DIR` | Checkpoint directory (`./checkpoints`) |
| `PORT` | FastAPI server port (`8000`) |
