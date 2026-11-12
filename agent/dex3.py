"""Dex3-1 hand kinematics & finger poses (open pre-grasp, C-hook curl, idle tuck)."""

from __future__ import annotations

import numpy as np

N_FINGER = 14
FINGER_KP = 8.0
FINGER_KD = 0.25

# Inner C-hook pockets in wrist_yaw_link local coordinates
HOOK_LOCAL_R = np.array([0.132, 0.032, 0.0], dtype=np.float64)
HOOK_LOCAL_L = np.array([0.132, -0.032, 0.0], dtype=np.float64)

# Safe tucked pose so arms never clip thighs when hanging
LEFT_TUCK = np.array([0.5, 0.7, 1.2, -1.2, -1.1, -1.2, -1.1], dtype=np.float32)
RIGHT_TUCK = np.array([0.5, -0.7, -1.2, 1.2, 1.1, 1.2, 1.1], dtype=np.float32)

# Pre-grasp open splay & active handle curl (respects XML joint limits per side)
OPEN_L = np.array([0.10, 0.10, 0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
CURL_L = np.array([0.0, 0.70, 1.20, -1.25, -1.45, -1.25, -1.45], dtype=np.float32)

OPEN_R = np.array([0.10, -0.10, -0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
CURL_R = np.array([0.0, -0.70, -1.20, 1.25, 1.45, 1.25, 1.45], dtype=np.float32)


def finger_pose(*, right_grip: float = 0.0, left_grip: float = 0.0, tuck_left: bool = True, tuck_right: bool = False) -> np.ndarray:
    """Return 14-D Dex3 target [left(7), right(7)]. grip in [0, 1]."""
    q = np.zeros(N_FINGER, dtype=np.float32)
    lg = float(np.clip(left_grip, 0.0, 1.0))
    rg = float(np.clip(right_grip, 0.0, 1.0))
    q[:7] = LEFT_TUCK if tuck_left else (1.0 - lg) * OPEN_L + lg * CURL_L
    q[7:] = RIGHT_TUCK if tuck_right else (1.0 - rg) * OPEN_R + rg * CURL_R
    return q
