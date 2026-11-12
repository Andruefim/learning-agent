"""Autonomous Unitree G1 fall-recovery / stand-up RL policy (SuperMimic-GetUp).

Runs the pre-trained 29-DoF whole-body tracking + fall-recovery policy from
RayZhao/supermimic_getup (Mr-Zqr/SuperMimic-Reproduced) directly in MuJoCo at
50 Hz with pure proprioception (zero teleportation or external forces).
"""

from __future__ import annotations

import collections
import math
import pathlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from agent.config import ROOT
from agent.h2 import N_ACT, STAND_Q

STANDUP_POLICY_PATH = ROOT / "models" / "unitree_g1" / "standup_supermimic.pt"

# Exact mjlab Unitree G1 motor / PD constants (asset_zoo/robots/unitree_g1/g1_constants.py)
def _reflected_inertia(ri: tuple[float, float, float], gr: tuple[float, float, float]) -> float:
    return ri[0] * (gr[1] * gr[2]) ** 2 + ri[1] * gr[2] ** 2 + ri[2]


_ARM_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1.0 + 46.0 / 18.0, 1.0 + 56.0 / 16.0))
_ARM_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1.0, 4.5, 1.0 + 48.0 / 22.0))
_ARM_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1.0, 4.5, 5.0))
_ARM_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))
_WN = 10.0 * 2.0 * math.pi
_ZETA = 2.0


def _motor_gains(armature: float, effort: float, mult: float = 1.0) -> tuple[float, float, float]:
    kp = armature * (_WN**2) * mult
    kd = 2.0 * _ZETA * armature * _WN * mult
    return kp, kd, effort * mult


_K_5020 = _motor_gains(_ARM_5020, 25.0)
_K_7520_14 = _motor_gains(_ARM_7520_14, 88.0)
_K_7520_22 = _motor_gains(_ARM_7520_22, 139.0)
_K_4010 = _motor_gains(_ARM_4010, 5.0)
_K_WAIST = _motor_gains(_ARM_5020, 25.0, 2.0)
_K_ANKLE = _motor_gains(_ARM_5020, 25.0, 2.0)

_JOINT_SPECS = [
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    _K_7520_14, _K_7520_22, _K_7520_14, _K_7520_22, _K_ANKLE, _K_ANKLE,
    # Right leg
    _K_7520_14, _K_7520_22, _K_7520_14, _K_7520_22, _K_ANKLE, _K_ANKLE,
    # Waist: yaw, roll, pitch
    _K_7520_14, _K_WAIST, _K_WAIST,
    # Left arm: sp, sr, sy, el, wr, wp, wy
    _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
    # Right arm
    _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
]

MJLAB_KP = np.array([g[0] for g in _JOINT_SPECS], dtype=np.float32)
MJLAB_KD = np.array([g[1] for g in _JOINT_SPECS], dtype=np.float32)
MJLAB_EFFORT = np.array([g[2] for g in _JOINT_SPECS], dtype=np.float32)
MJLAB_ACTION_SCALE = (0.25 * MJLAB_EFFORT / MJLAB_KP).astype(np.float32)

# KNEES_BENT_KEYFRAME (mjlab default joint offsets q0)
MJLAB_Q0 = np.zeros(N_ACT, dtype=np.float32)
for _i in (0, 6):
    MJLAB_Q0[_i] = -0.312
for _i in (3, 9):
    MJLAB_Q0[_i] = 0.669
for _i in (4, 10):
    MJLAB_Q0[_i] = -0.363
for _i in (15, 22):
    MJLAB_Q0[_i] = 0.2
for _i in (18, 25):
    MJLAB_Q0[_i] = 0.6
MJLAB_Q0[16] = 0.2
MJLAB_Q0[23] = -0.2

# Upright stand target matching System 0 legs/arms
_Q_STAND_REF = np.zeros(N_ACT, dtype=np.float32)
for _i in (0, 6):
    _Q_STAND_REF[_i] = -0.10
for _i in (3, 9):
    _Q_STAND_REF[_i] = 0.30
for _i in (4, 10):
    _Q_STAND_REF[_i] = -0.20
for _i in (15, 22):
    _Q_STAND_REF[_i] = 0.25
for _i in (18, 25):
    _Q_STAND_REF[_i] = 1.28
_Q_STAND_REF[16] = 0.20
_Q_STAND_REF[23] = -0.20


def _proj_gravity_wxyz(qwxyz: np.ndarray) -> np.ndarray:
    """Gravity vector (0, 0, -1) expressed in body/pelvis frame."""
    qw, qx, qy, qz = (float(qwxyz[0]), float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3]))
    return np.array(
        [
            -2.0 * (qx * qz - qw * qy),
            -2.0 * (qy * qz + qw * qx),
            -(1.0 - 2.0 * (qx * qx + qy * qy)),
        ],
        dtype=np.float32,
    )


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MotionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP(680, (2048, 1024, 512, 256), 32)
        self.register_buffer("scale", torch.ones(680))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP(962, (2048, 1024, 512, 256), N_ACT)
        self.register_buffer("scale", torch.ones(930))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class _SuperMimicNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({"motion": _MotionEncoder()})
        self.decoder = _Decoder()

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoders["motion"](goal), obs_hist)


def _load_ckpt_windows_safe(path: Path) -> dict:
    orig_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = orig_posix


class G1StandUpPolicy:
    """50 Hz fall-recovery controller using SuperMimic whole-body get-up weights."""

    def __init__(self, path: Path = STANDUP_POLICY_PATH) -> None:
        ckpt = _load_ckpt_windows_safe(path)
        self.net = _SuperMimicNet()
        self.net.load_state_dict(ckpt["policy"], strict=True)
        self.net.eval()

        # Precompute static standing goal tensor (10 future frames x 68 features)
        g1 = np.zeros(68, dtype=np.float32)
        g1[0:N_ACT] = _Q_STAND_REF - MJLAB_Q0
        g1[58:61] = [0.0, 0.0, -1.0]
        g1[67] = 0.7837
        self._goal_t = torch.from_numpy(np.tile(g1, 10)).unsqueeze(0)

        self._hist: collections.deque[np.ndarray] = collections.deque(maxlen=10)
        self.last_action = np.zeros(N_ACT, dtype=np.float32)
        self.target_q = MJLAB_Q0.copy()

    def _obs_step(self, data, qadr: np.ndarray, vadr: np.ndarray) -> np.ndarray:
        q = np.asarray(data.qpos[qadr[:N_ACT]], dtype=np.float32)
        qd = np.asarray(data.qvel[vadr[:N_ACT]], dtype=np.float32)
        ang = np.asarray(data.qvel[3:6], dtype=np.float32)
        grav = _proj_gravity_wxyz(data.qpos[3:7])
        return np.concatenate([q - MJLAB_Q0, qd, ang, grav, self.last_action]).astype(np.float32)

    def reset(self, data=None, qadr: Optional[np.ndarray] = None, vadr: Optional[np.ndarray] = None) -> None:
        self.last_action[:] = 0.0
        self.target_q[:] = MJLAB_Q0
        self._hist.clear()
        if data is not None and qadr is not None and vadr is not None:
            o0 = self._obs_step(data, qadr, vadr)
        else:
            o0 = np.zeros(93, dtype=np.float32)
            o0[61:64] = [0.0, 0.0, -1.0]
        for _ in range(10):
            self._hist.append(o0.copy())

    def update(self, data, qadr: np.ndarray, vadr: np.ndarray) -> None:
        """One 50 Hz policy step (called every DECIMATION = 10 MuJoCo ticks)."""
        if len(self._hist) < 10:
            self.reset(data, qadr, vadr)
        else:
            self._hist.append(self._obs_step(data, qadr, vadr))

        h = np.stack(list(self._hist), axis=0)  # (10, 93)
        # Flatten in mjlab term-major layout: [joint_pos(290), joint_vel(290), ang(30), grav(30), act(290)]
        obs_flat = np.concatenate(
            [
                h[:, 0:29].reshape(-1),
                h[:, 29:58].reshape(-1),
                h[:, 58:61].reshape(-1),
                h[:, 61:64].reshape(-1),
                h[:, 64:93].reshape(-1),
            ]
        )
        with torch.no_grad():
            act = self.net(torch.from_numpy(obs_flat).unsqueeze(0), self._goal_t).squeeze(0).numpy()
        self.last_action[:] = act.astype(np.float32)
        self.target_q[:] = MJLAB_Q0 + self.last_action * MJLAB_ACTION_SCALE

    def torques(
        self,
        data,
        qadr: np.ndarray,
        vadr: np.ndarray,
        stand_kp: np.ndarray,
        stand_kd: np.ndarray,
        blend_w: float = 0.0,
    ) -> np.ndarray:
        """Compute 29-DoF motor torques, optionally blending smoothly into System 0 STAND_Q."""
        q = np.asarray(data.qpos[qadr[:N_ACT]], dtype=np.float32)
        qd = np.asarray(data.qvel[vadr[:N_ACT]], dtype=np.float32)
        w = float(np.clip(blend_w, 0.0, 1.0))
        if w <= 0.0:
            tau = (self.target_q - q) * MJLAB_KP - qd * MJLAB_KD
        else:
            q_des = (1.0 - w) * self.target_q + w * STAND_Q
            kp = (1.0 - w) * MJLAB_KP + w * np.asarray(stand_kp[:N_ACT], dtype=np.float32)
            kd = (1.0 - w) * MJLAB_KD + w * np.asarray(stand_kd[:N_ACT], dtype=np.float32)
            bias = np.asarray(data.qfrc_bias[vadr[:N_ACT]], dtype=np.float32)
            tau = (q_des - q) * kp - qd * kd + w * bias
        return np.clip(tau, -MJLAB_EFFORT, MJLAB_EFFORT)


def load_g1_standup() -> G1StandUpPolicy | None:
    if not STANDUP_POLICY_PATH.is_file():
        return None
    return G1StandUpPolicy(STANDUP_POLICY_PATH)
