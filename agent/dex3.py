"""Dex3-1 fingers. They sit after the 29 body actuators and do not enter the walk net.

Order matches g1_29dof_dex3.xml: thumb, index, middle, left then right.
Zero is the open hand. Curl is the flexion limit, scaled so the tips meet.
"""

from __future__ import annotations

import numpy as np

N_FINGER = 14
FINGER_KP = 4.0
FINGER_KD = 0.15

# thumb0, thumb1, thumb2, index0, index1, middle0, middle1
_CLOSE_LEFT = np.array([0.5, 0.7, 1.2, -1.2, -1.1, -1.2, -1.1], dtype=np.float32)
_CLOSE_RIGHT = np.array([0.5, -0.7, -1.2, 1.2, 1.1, 1.2, 1.1], dtype=np.float32)


def finger_target(goal: dict | None) -> np.ndarray:
    """Open unless a world point is already in the hand's reach phase."""
    q = np.zeros(N_FINGER, dtype=np.float32)
    if not goal or goal.get("point") is None or goal.get("target") == "head":
        return q
    hand = str(goal.get("hand") or "")
    if hand in ("left", "both"):
        q[:7] = _CLOSE_LEFT
    if hand in ("right", "both"):
        q[7:] = _CLOSE_RIGHT
    return q
