"""G1 backflip from the pretrained wbc-mjlab tracker.

The network is the ONNX policy in models/g1_flip. It does not invent the
motion: it tracks flip_02.npz, a flip clip shipped with that policy.
Observation is 132-D, action is a residual on the clip's joint pose.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FLIP_DIR = ROOT / "models" / "g1_flip"
POLICY_DT = 0.02
ANCHOR = 15  # torso_link in the 30-body G1 clip (no Dex3 fingers)
PELVIS = 0


def wants_flip(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ("сальто", "сально", "backflip", "бэкфлип", "salto"))


def _quat_apply_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    qc = np.zeros(4, np.float64)
    out = np.zeros(3, np.float64)
    mujoco.mju_negQuat(qc, np.asarray(q_wxyz, np.float64))
    mujoco.mju_rotVecQuat(out, np.asarray(v, np.float64), qc)
    return out.astype(np.float32)


class FlipPolicy:
    def __init__(self, directory: Path = FLIP_DIR, clip: str = "flip_01.npz"):
        import onnxruntime as ort

        cfg = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        clip = np.load(directory / clip, allow_pickle=True)
        self.default = np.asarray(cfg["default_joint_pos"], np.float32)
        self.kp = np.asarray(cfg["stiffness"], np.float32)
        self.kd = np.asarray(cfg["damping"], np.float32)
        self.scale = np.asarray(cfg["action"]["scale"], np.float32)
        self.joint_pos = np.asarray(clip["joint_pos"], np.float32)
        self.body_pos = np.asarray(clip["body_pos_w"], np.float32)
        self.body_quat = np.asarray(clip["body_quat_w"], np.float32)
        self.body_lin = np.asarray(clip["body_lin_vel_w"], np.float32)
        self.body_ang = np.asarray(clip["body_ang_vel_w"], np.float32)
        self.n = int(self.joint_pos.shape[0])
        self.session = ort.InferenceSession(
            str(directory / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        self.frame = 0
        self.last_a = np.zeros(29, np.float32)

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, qadr: np.ndarray, xy: tuple[float, float] | None = None) -> None:
        """Place the clip's first frame at `xy`, keeping the robot's current facing.

        The clip's root quaternion is the mocap performer's heading, not the robot's.
        """
        self.frame = 0
        self.last_a[:] = 0
        facing = np.array(data.qpos[3:7], dtype=np.float64).copy()
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        pos = self.body_pos[0, PELVIS].copy()
        if xy is not None:
            pos[0], pos[1] = xy
        data.qpos[0:3] = pos
        data.qpos[3:7] = facing
        data.qpos[qadr] = self.joint_pos[0]
        mujoco.mj_forward(model, data)

    def _reference(self) -> np.ndarray:
        i = self.frame
        q = self.body_quat[i, ANCHOR]
        return np.concatenate(
            [
                self.body_pos[i, ANCHOR, 2:3],
                _quat_apply_inv(q, self.body_lin[i, ANCHOR]),
                _quat_apply_inv(q, self.body_ang[i, ANCHOR]),
                _quat_apply_inv(q, np.array([0.0, 0.0, -1.0], np.float32)),
                self.joint_pos[i],
            ]
        ).astype(np.float32)

    def _obs(self, model: mujoco.MjModel, data: mujoco.MjData, qadr: np.ndarray, vadr: np.ndarray) -> np.ndarray:
        pelvis = model.body("pelvis").id
        R = data.xmat[pelvis].reshape(3, 3)
        vel = np.zeros(6, np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, vel, 1)
        q = np.asarray(data.qpos[qadr], np.float32)
        qd = np.asarray(data.qvel[vadr], np.float32)
        return np.concatenate(
            [
                self._reference(),
                vel[:3].astype(np.float32),
                (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32),
                q - self.default,
                qd,
                self.last_a,
            ]
        ).astype(np.float32)

    def act(self, model: mujoco.MjModel, data: mujoco.MjData, qadr: np.ndarray, vadr: np.ndarray) -> np.ndarray:
        obs = self._obs(model, data, qadr, vadr)[None]
        raw = self.session.run(["actions"], {"obs": obs})[0][0].astype(np.float32)
        self.last_a = raw
        target = self.scale * raw + self.joint_pos[self.frame]
        self.frame = min(self.frame + 1, self.n - 1)
        return target

    @property
    def done(self) -> bool:
        return self.frame >= self.n - 1


def flip_torque(model: mujoco.MjModel, data: mujoco.MjData, q_target: np.ndarray, policy: FlipPolicy, qadr: np.ndarray, vadr: np.ndarray) -> np.ndarray:
    q = np.asarray(data.qpos[qadr], np.float32)
    qd = np.asarray(data.qvel[vadr], np.float32)
    tau = policy.kp * (q_target - q) - policy.kd * qd
    n = tau.shape[0]
    lo = np.asarray(model.actuator_ctrlrange[:n, 0], np.float32)
    hi = np.asarray(model.actuator_ctrlrange[:n, 1], np.float32)
    return np.clip(tau, lo, hi)
