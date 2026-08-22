"""Level 3 conditioned neural balance at the physics rate (S0 / 200 Hz).

12-DoF leg residual + 14-DoF arm conditioning from L2. Last layer can start
at zero so an untrained net is a no-op on the nominal stance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agent.h2 import STAND_Q, SQUAT_Q

LEG_DOF = 12
UPPER_DOF = 14
LEG_IDX = tuple(range(0, 12))
UPPER_IDX = tuple(range(15, 29))
OBS_DIM = 3 + 3 + LEG_DOF + LEG_DOF + (UPPER_DOF + 2)
ACT_DIM = LEG_DOF
L3_DELTA_LIM = 0.45
# Back-compat aliases for tests.
LEG_JOINTS = LEG_IDX
UPPER_JOINTS = UPPER_IDX


class Level3BalancePolicy(nn.Module):
    def __init__(self, leg_dof: int = LEG_DOF, upper_dof: int = UPPER_DOF, *, zero_out: bool = True):
        super().__init__()
        in_dim = 3 + 3 + leg_dof + leg_dof + (upper_dof + 2)
        self.leg_dof = int(leg_dof)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, leg_dof),
        )
        if zero_out:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def deltas(self, obs: torch.Tensor) -> torch.Tensor:
        return L3_DELTA_LIM * torch.tanh(self.forward(obs))


def torso_imu(data, torso_id: int) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    grav = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gyro = np.asarray(data.cvel[torso_id, :3], dtype=np.float64)
    return grav.astype(np.float32), gyro.astype(np.float32)


def legs_nominal(height_01: float) -> np.ndarray:
    h = float(np.clip(height_01, 0.0, 1.0))
    return ((1.0 - h) * SQUAT_Q + h * STAND_Q).astype(np.float32)


def build_obs(
    data,
    torso_id: int,
    q: np.ndarray,
    qd: np.ndarray,
    q_upper_target: np.ndarray,
    target_height: float,
    target_vx: float,
) -> np.ndarray:
    """IMU + leg error/vel vs stand + L2 cond [z_pelvis_m, vx, 14 arm targets]."""
    grav, gyro = torso_imu(data, torso_id)
    q = np.asarray(q, dtype=np.float32)
    qd = np.asarray(qd, dtype=np.float32)
    qerr = q[list(LEG_IDX)] - STAND_Q[list(LEG_IDX)]
    qvel = qd[list(LEG_IDX)]
    upper = np.asarray(q_upper_target, dtype=np.float32).reshape(-1)[list(UPPER_IDX)]
    cond = np.concatenate(
        [
            np.array([float(target_height), float(target_vx)], dtype=np.float32),
            upper,
        ]
    )
    return np.concatenate([grav, gyro, qerr, qvel, cond]).astype(np.float32)


def apply_leg_deltas(q: np.ndarray, delta: np.ndarray, height_01: float = 1.0) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    d = np.clip(np.asarray(delta, dtype=np.float32).reshape(-1), -L3_DELTA_LIM, L3_DELTA_LIM)
    if d.shape[0] != LEG_DOF:
        raise ValueError(f"L3 delta dim {d.shape[0]} != {LEG_DOF}")
    nom = legs_nominal(height_01)
    for i, j in enumerate(LEG_IDX):
        out[j] = nom[j] + d[i]
    return out
