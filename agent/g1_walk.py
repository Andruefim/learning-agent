"""Official Unitree G1 leg policy (deploy/pre_train/g1/motion.pt).

12 leg joints, 47-D observation, 50 Hz. PD gains and scales match
deploy/deploy_mujoco/configs/g1.yaml. No gravity compensation: the network
was trained with that torque law.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agent.config import ROOT

POLICY_PATH = ROOT / "models" / "unitree_g1" / "motion.pt"
N_LEG = 12
N_OBS = 47
PERIOD = 0.8
POLICY_DT = 0.02  # 50 Hz, matches DECIMATION * 0.002 s
CMD_IDLE = 0.05
# After the velocity command drops, shrink the gait and hand the legs to the
# stiff pose PD. Cutting the clock in one step, or freezing it, either keeps
# stepping or falls: this net has no static stand point.
FADE_SEC = 1.0
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32,
)
LEG_KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
LEG_KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Projected gravity. quaternion is MuJoCo wxyz."""
    qw, qx, qy, qz = (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class G1LegPolicy:
    """Official walk net for the 12 leg joints.

    At a zero command the net still steps, because its phase clock is the gait.
    Idle therefore does not use it: the stiff pose PD holds the stand. Coming
    out of a step, the gait amplitude is faded down while that PD takes over,
    so the robot neither keeps marching nor falls out of the swing pose.
    """

    def __init__(self, path: Path = POLICY_PATH):
        self.policy = torch.jit.load(str(path), map_location="cpu")
        self.policy.eval()
        self.action = np.zeros(N_LEG, dtype=np.float32)
        self.target = DEFAULT_ANGLES.copy()
        self._phase_t = 0.0
        self._fade = 1.0

    def reset(self) -> None:
        self.action[:] = 0.0
        self.target[:] = DEFAULT_ANGLES
        self._phase_t = 0.0
        self._fade = 1.0

    @property
    def holding(self) -> bool:
        """True once the legs are on the stiff stand PD and the net is idle."""
        return self._fade >= 1.0

    def update(self, data, qadr: np.ndarray, vadr: np.ndarray, cmd_v: np.ndarray, *, dynamic: bool = False) -> None:
        """One 50 Hz step. cmd_v is [vx, vy, wz] in the same units as Unitree's yaml.

        dynamic keeps the clock running at a zero velocity command, so a
        forward arm pose can be balanced by stepping instead of a fixed stand.
        """
        cmd = np.asarray(cmd_v, dtype=np.float32).reshape(3)
        moving = float(np.max(np.abs(cmd))) > CMD_IDLE
        active = moving or dynamic
        if not active and self._fade >= 1.0:
            return
        if not moving:
            cmd = np.zeros(3, dtype=np.float32)
        self._phase_t += POLICY_DT
        q = np.asarray(data.qpos[qadr[:N_LEG]], dtype=np.float32)
        dq = np.asarray(data.qvel[vadr[:N_LEG]], dtype=np.float32)
        obs = np.zeros(N_OBS, dtype=np.float32)
        obs[0:3] = np.asarray(data.qvel[3:6], dtype=np.float32) * ANG_VEL_SCALE
        obs[3:6] = gravity_orientation(data.qpos[3:7])
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = (q - DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[21:33] = dq * DOF_VEL_SCALE
        obs[33:45] = self.action
        phase = (self._phase_t % PERIOD) / PERIOD
        obs[45] = np.sin(2.0 * np.pi * phase)
        obs[46] = np.cos(2.0 * np.pi * phase)
        with torch.no_grad():
            out = self.policy(torch.from_numpy(obs).unsqueeze(0))
        self.action[:] = out.detach().cpu().numpy().reshape(N_LEG)
        self.target[:] = self.action * ACTION_SCALE + DEFAULT_ANGLES

    def leg_torque(
        self,
        data,
        qadr: np.ndarray,
        vadr: np.ndarray,
        cmd_v: np.ndarray,
        stiff_kp: np.ndarray,
        stiff_kd: np.ndarray,
        dt: float,
        *,
        dynamic: bool = False,
    ) -> np.ndarray | None:
        """Soft gait torque, a fading blend, or None when the stiff pose PD should run."""
        cmd = np.asarray(cmd_v, dtype=np.float32).reshape(3)
        moving = float(np.max(np.abs(cmd))) > CMD_IDLE or dynamic
        q = np.asarray(data.qpos[qadr[:N_LEG]], dtype=np.float32)
        dq = np.asarray(data.qvel[vadr[:N_LEG]], dtype=np.float32)
        if moving:
            self._fade = 0.0
            return (self.target - q) * LEG_KP - dq * LEG_KD
        if self._fade >= 1.0:
            return None
        self._fade = min(1.0, self._fade + float(dt) / FADE_SEC)
        if self._fade >= 1.0:
            return None
        weight = 1.0 - self._fade
        target = DEFAULT_ANGLES + self.action * ACTION_SCALE * weight
        kp = weight * LEG_KP + self._fade * np.asarray(stiff_kp[:N_LEG], dtype=np.float32)
        kd = weight * LEG_KD + self._fade * np.asarray(stiff_kd[:N_LEG], dtype=np.float32)
        bias = np.asarray(data.qfrc_bias[vadr[:N_LEG]], dtype=np.float32)
        return (target - q) * kp - dq * kd + self._fade * bias


def load_g1_walk() -> G1LegPolicy | None:
    if not POLICY_PATH.is_file():
        return None
    return G1LegPolicy(POLICY_PATH)
