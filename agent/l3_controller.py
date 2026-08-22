"""Level 3 conditioned neural balance. Runs at the physics rate (S0 / 200 Hz).

Outputs residual target-angle deltas for the 10 balance legs (hip pitch/roll,
knee, ankle pitch/roll). Last layer starts at zero so an untrained net is a
no-op on top of the tracker PD. Official H2 MJCF is torque-driven; this module
edits q_cmd, not XML actuators.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agent.h2 import L_AK, L_AZ, L_EL, L_HX, L_HY, L_KN, L_SH, L_SX, R_AK, R_AZ, R_EL, R_HX, R_HY, R_KN, R_SH, R_SX

OBS_DIM = 26
ACT_DIM = 10
L3_DELTA_LIM = 0.35

# Hip pitch/roll, knee, ankle pitch/roll. Hip yaw stays with the turn tracker.
LEG_JOINTS = (R_HY, L_HY, R_HX, L_HX, R_KN, L_KN, R_AK, L_AK, R_AZ, L_AZ)
UPPER_JOINTS = (R_SH, L_SH, R_EL, L_EL, R_SX, L_SX)


class Level3BalancePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs (26 dim):
        # - IMU Gravity Vector (3): [0, 0, -1] in torso coordinates
        # - IMU Angular Velocity (3): torso gyro (wx, wy, wz)
        # - Legs Proprioception (10): current angles of LEG_JOINTS
        # - L2 Conditioning (10): [target_height, target_vx, target_upper_body_q(6), target_yaw(2)]
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, ACT_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def deltas(self, x: torch.Tensor) -> torch.Tensor:
        return L3_DELTA_LIM * torch.tanh(self.forward(x))


def torso_imu(data, torso_id: int) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    grav = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gyro = np.asarray(data.cvel[torso_id, :3], dtype=np.float64)
    return grav.astype(np.float32), gyro.astype(np.float32)


def l2_conditioning(q_des: np.ndarray, height: float, vx: float, yaw: float) -> np.ndarray:
    upper = np.asarray(q_des, dtype=np.float32)[list(UPPER_JOINTS)]
    yaw = float(yaw)
    return np.concatenate(
        [
            np.array([float(height), float(vx)], dtype=np.float32),
            upper,
            np.array([np.sin(np.pi * yaw), np.cos(np.pi * yaw)], dtype=np.float32),
        ]
    )


def build_obs(
    data,
    torso_id: int,
    q: np.ndarray,
    q_des: np.ndarray,
    height: float,
    vx: float,
    yaw: float,
) -> np.ndarray:
    grav, gyro = torso_imu(data, torso_id)
    legs = np.asarray(q, dtype=np.float32)[list(LEG_JOINTS)]
    cond = l2_conditioning(q_des, height, vx, yaw)
    return np.concatenate([grav, gyro, legs, cond]).astype(np.float32)


def apply_leg_deltas(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    d = np.clip(np.asarray(delta, dtype=np.float32).reshape(-1), -L3_DELTA_LIM, L3_DELTA_LIM)
    if d.shape[0] != len(LEG_JOINTS):
        raise ValueError(f"L3 delta dim {d.shape[0]} != {len(LEG_JOINTS)}")
    for i, j in enumerate(LEG_JOINTS):
        out[j] = out[j] + d[i]
    return out
