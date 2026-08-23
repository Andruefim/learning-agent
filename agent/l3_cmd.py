"""L1/L2 → Level-3 command. No CPG: velocity, height, arm targets only."""

from __future__ import annotations

import numpy as np

from agent.h2 import ACTION_DIM, ARM_RAISE
from agent.plan import Plan, TeacherIntent, parse_requested_yaw

# vx, vy, wz, h_m, 14 arm joints (actuator order, both arms)
L2_CMD_DIM = ACTION_DIM
ARM_DOF = 14
UPPER_IDX = tuple(range(15, 29))
CMD_VX, CMD_VY, CMD_WZ, CMD_H = 0, 1, 2, 3
CMD_ARMS = slice(4, 18)


def arm_targets(t: TeacherIntent) -> np.ndarray:
    q = np.zeros(ARM_DOF, dtype=np.float32)
    q[0] = float(ARM_RAISE) * float(np.clip(t.l_arm, 0.0, 1.0))
    q[7] = float(ARM_RAISE) * float(np.clip(t.r_arm, 0.0, 1.0))
    q[1] = 1.20 * float(np.clip(t.l_out, -1.0, 1.0))
    q[8] = -1.20 * float(np.clip(t.r_out, -1.0, 1.0))
    q[3] = 0.90 * float(np.clip(t.l_arm, 0.0, 1.0))
    q[10] = 0.90 * float(np.clip(t.r_arm, 0.0, 1.0))
    return q


def height_m(height_01: float) -> float:
    h = float(np.clip(height_01, 0.0, 1.0))
    return float(0.62 + h * (1.02 - 0.62))


def command_from_intent(t: TeacherIntent, *, requested_yaw: float | None = None) -> np.ndarray:
    cmd = np.zeros(L2_CMD_DIM, dtype=np.float32)
    cmd[CMD_VX] = float(np.clip(t.vx, -1.2, 1.2))
    cmd[CMD_VY] = 0.0
    wz = float(np.clip(t.yaw, -1.0, 1.0)) * 0.7
    if requested_yaw is not None:
        wz = float(np.clip(np.sign(requested_yaw) * min(abs(float(requested_yaw)), 1.0), -1.0, 1.0))
    cmd[CMD_WZ] = wz
    cmd[CMD_H] = height_m(t.height)
    cmd[CMD_ARMS] = arm_targets(t)
    return cmd


def command_from_plan(plan: Plan, *, exec_bias: dict | None = None) -> np.ndarray:
    t = plan.teacher()
    cmd = command_from_intent(t, requested_yaw=parse_requested_yaw(plan.skill, plan.params))
    bias = exec_bias or {}
    cmd[CMD_VX] = float(np.clip(cmd[CMD_VX] + float(bias.get("vx", 0.0)), -1.2, 1.2))
    cmd[CMD_H] = float(np.clip(cmd[CMD_H] + float(bias.get("h", 0.0)) * 0.2, 0.60, 1.02))
    cmd[CMD_WZ] = float(np.clip(cmd[CMD_WZ] + float(bias.get("yaw", 0.0)) * 0.5, -1.0, 1.0))
    return cmd


def clip_command(cmd: np.ndarray) -> np.ndarray:
    out = np.asarray(cmd, dtype=np.float32).reshape(L2_CMD_DIM).copy()
    out[CMD_VX] = float(np.clip(out[CMD_VX], -1.2, 1.2))
    out[CMD_VY] = float(np.clip(out[CMD_VY], -0.4, 0.4))
    out[CMD_WZ] = float(np.clip(out[CMD_WZ], -1.0, 1.0))
    out[CMD_H] = float(np.clip(out[CMD_H], 0.60, 1.02))
    return out


def stand_command() -> np.ndarray:
    cmd = np.zeros(L2_CMD_DIM, dtype=np.float32)
    cmd[CMD_H] = 1.02
    return cmd


def reach_command(*, pitch: float = 1.0, asymmetric: bool = False) -> np.ndarray:
    """Arms forward. pitch=1 → 90° (ARM_RAISE)."""
    cmd = stand_command()
    p = float(np.clip(pitch, 0.0, 1.0))
    t = TeacherIntent(r_arm=p, l_arm=0.35 if asymmetric else p)
    cmd[CMD_ARMS] = arm_targets(t)
    return cmd


def deep_squat_command(*, h_m: float = 0.70) -> np.ndarray:
    cmd = stand_command()
    cmd[CMD_H] = float(np.clip(h_m, 0.60, 1.02))
    return cmd
