"""Shared constants. Body-specific joints and poses live in `agent.h2`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"

ROOT = Path(__file__).resolve().parent.parent
ERROR_LEN = 8
CHUNK = 12
FLOW_STEPS = 2
VISION_H, VISION_W = 48, 64
VISION_STRIDE = 4
VISION_DIM = 64
INSTR_BYTES = 64
LANG_DIM = 32
Z_DIM = 8
L1_PERIOD = 2.5
SLEW = 0.04
TRACK_OK = 0.05
BRACE_TILT = 0.72
FALL_Z = 0.50
STAND_Z = 1.02
BRACE_ERR_WIN = 80
BRACE_TILT_WIN = 40
TURN_IN_CONTEXT_OK = False
TRIAL_MAX = 3
TRIAL_EMB = 32
PARAM_KEYS = ("h", "vx", "yaw", "steps", "wave", "kick", "r_arm", "l_arm")
SKILL_IDS = ("hold", "stand", "squat", "locomote", "turn", "wave", "kick", "reach")
SKILL_TO_I = {name: i for i, name in enumerate(SKILL_IDS)}
DEFAULT_PARAMS = {
    "h": 1.0,
    "vx": 0.0,
    "yaw": 0.0,
    "steps": 0.0,
    "wave": 0.0,
    "kick": 0.0,
    "r_arm": 0.0,
    "l_arm": 0.0,
}
PARAM_SCALE = {
    "h": 1.0,
    "vx": 1.0,
    "yaw": 1.0,
    "steps": 8.0,
    "wave": 1.0,
    "kick": 1.0,
    "r_arm": 1.0,
    "l_arm": 1.0,
}
PARAM_OK = {
    "h": 0.08,
    "vx": 0.22,
    "yaw": 0.22,
    "steps": 0.20,
    "wave": 0.45,
    "kick": 0.40,
    "r_arm": 0.22,
    "l_arm": 0.22,
}
PARAM_MEASURE = {
    "h": "pelvis height / stand height",
    "vx": "forward displacement over the chunk",
    "yaw": "pelvis yaw change",
    "steps": "foot-contact step count",
    "wave": "arm oscillation completed",
    "kick": "leg flexion profile completed",
    "r_arm": "right arm raise 0..1",
    "l_arm": "left arm raise 0..1",
}
SHADOW_MSE_MAX = 0.12
SHADOW_EMA = 0.95
SHADOW_EMA_TICKS = 200
REPLAY_GATE = 1000
ALPHA_START = 0.03
ALPHA_STEP = 0.02
# Bounded Residual (Variant A): student adds a clipped residual on the L3 command.
# ALPHA_MAX is the standing design, not a demo knob. Do not raise it without a
# new stability argument. Stage D (Safety-Shield, alpha→1 with emergency
# takeover) is a possible future direction — not implemented. If added, the
# shield must trip on fall trends (tilt/z), not late static thresholds.
ALPHA_MAX = 0.12
RESIDUAL_LIMIT = 0.25
ROLLBACK_FALL_DELTA = 0.08
# Legacy CPG constants (unused by the foundation controller).
PIVOT_HX_SHIFT = 0.45
PIVOT_HX_OPEN = 0.22
HX_PD_CLIP = 0.25
HX_PD_CLIP_UNLOAD = 0.06
H1_SPEC = {
    "kind": "offline_heldout_plus_live_shadow",
    "requires_student_on_actuators": False,
    "offline": (
        "Held-out teacher command chunks from replay. Student forward only; physics untouched. "
        "Pass if MSE(true error deque) < 0.05 and beats zero-deque and shuffled-error ablation by 1e-4."
    ),
    "shadow": (
        "Every vision tick: u_student vs l3_cmd from L1, actuators stay on the foundation+PD. "
        f"Stage B after shadow_mse_ema < {SHADOW_MSE_MAX} for {SHADOW_EMA_TICKS} ticks, "
        f"replay>={REPLAY_GATE}, and fall rate not rising. "
        "Bounded Residual ALPHA_MAX=0.12 (Variant A) on the 18-DoF command, not joint angles."
    ),
    "shadow_streak": SHADOW_EMA_TICKS,
    "shadow_mse_max": SHADOW_MSE_MAX,
    "replay_gate": REPLAY_GATE,
    "residual": (
        "cmd = l3_cmd + alpha * clip(u_student - l3_cmd, +/-limit). "
        "alpha starts at 0.03, grows only while ema stays in band; "
        "auto-rollback to last working alpha if fall rate rises. "
        "ALPHA_MAX=0.12 is the cap; Joint-PD tracks the foundation q_target."
    ),
    "stage_d": (
        "Future Safety-Shield only: alpha->1 with takeover on fall trend (tilt/z). "
        "Static tilt<0.8 / e>4*TRACK_OK are too late."
    ),
    "weights": "Updated only on Save/Consolidate (offline CFM). Runtime eval(), 0 backprop.",
}
