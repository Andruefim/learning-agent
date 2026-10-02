"""Dex3-1 fingers. They sit after the 29 body actuators and do not enter the walk net.

Order matches g1_29dof_dex3.xml: thumb, index, middle, left then right.
Zero is a flat open hand (thumbs stick out into the hips). Curl tucks them
so the arm can raise past the pelvis.
"""

from __future__ import annotations

import numpy as np

N_FINGER = 14
FINGER_KP = 4.0
FINGER_KD = 0.15

# thumb0, thumb1, thumb2, index0, index1, middle0, middle1
_CLOSE_LEFT = np.array([0.5, 0.7, 1.2, -1.2, -1.1, -1.2, -1.1], dtype=np.float32)
_CLOSE_RIGHT = np.array([0.5, -0.7, -1.2, 1.2, 1.1, 1.2, 1.1], dtype=np.float32)


def finger_curl(hand: str = "both") -> np.ndarray:
    """Tucked fingers — safe to raise past the thighs."""
    q = np.zeros(N_FINGER, dtype=np.float32)
    side = str(hand or "both")
    if side in ("left", "both"):
        q[:7] = _CLOSE_LEFT
    if side in ("right", "both"):
        q[7:] = _CLOSE_RIGHT
    return q


def finger_target(goal: dict | None) -> np.ndarray:
    """Curl the named hand when a grasp goal is active; otherwise stay tucked.

    A flat open hand is only useful for a brief pre-grasp, and it collides with
    the hips on the way up — so the default is curl, not splay.
    """
    if not goal or goal.get("point") is None or goal.get("target") == "head":
        return finger_curl("both")
    return finger_curl(str(goal.get("hand") or "both"))
