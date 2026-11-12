"""Shared configuration for the SLEEP continual-learning Unitree G1 runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform in ("darwin", "win32") else "egl"

CHECKPOINT_DIR = Path(os.getenv("STORAGE_DIR", str(ROOT / "checkpoints")))

# Physics & camera dimensions
PHYSICS_DT = 0.002
COGNITIVE_DT = 0.10          # 10 Hz System 1 / System 2 chunk execution tick
SUBSTEPS_PER_COG_STEP = int(round(COGNITIVE_DT / PHYSICS_DT))  # 50 steps = 0.1s
DECIMATION = 10              # 50 Hz RL gait update (0.02 s)

LOOK_W, LOOK_H = 640, 480    # Head camera RGB & depth for VLM grounding
PREVIEW_W, PREVIEW_H = 1280, 720
PREVIEW_INTERVAL = 1.0 / 30.0

# Balance / fall safety thresholds
FALL_Z = 0.48
TILT_LIM = 0.55
STAND_Z = 0.783
SLEW_RAD = 0.06

# SLEEP Continual Learning (arXiv:2606.03979)
SLEEP_IMG_SIZE = 128
SLEEP_HORIZON = 6
SLEEP_ACTION_DIM = 20        # [vx, wz, squat, waist_y, waist_p, r_arm(7), l_arm(7), r_grip]
SLEEP_PROPRIO_DIM = 28
TAU_THINK_DEFAULT = 0.35
MICRO_SLEEP_IDLE_S = 3.5
MICRO_SLEEP_MIN_STEPS = 8
MAX_TASK_VOCAB = 32
