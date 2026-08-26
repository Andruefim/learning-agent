"""Native MuJoCo foundation env: 50 Hz policy, 200 Hz Joint-PD, curriculum rewards.

Used for stand eval and as the CPU fallback when MJX-JAX is unavailable.
The MJX twin lives in `agent/l3_mjx.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco

from agent.h2 import (
    N_ACT,
    SPAWN_Z,
    STAND_Q,
    TRAIN_XML,
    actuator_addrs,
    box_geom,
    colliding_geoms,
    cylinders_to_capsules,
    disable_foot_spheres,
    disable_mesh_contacts,
    joint_limits,
)
from agent.joint_pd import compute_torques, kp_kd_vectors
from agent.l3_cmd import (
    CMD_ARMS,
    CMD_H,
    CMD_VX,
    CMD_VY,
    CMD_WZ,
    L2_CMD_DIM,
    arm_targets,
    stand_command,
)
from agent.plan import TeacherIntent
from agent.l3_foundation import (
    ACT_DIM,
    DECIMATION,
    EPISODE_SEC,
    HEIGHT_RANGE,
    OBS_DIM,
    PUSH_DUR_SEC,
    PUSH_EVERY_SEC,
    PUSH_FORCE,
    REACH_FRAC,
    REWARD_CLIP,
    STAND_ONLY,
    TERMINAL_PENALTY,
    TRAIN_FALL_Z,
    TRAIN_TILT,
    balance_delta,
    body_xy,
    build_obs,
    com_err_xy,
    foot_pitch_from_xmat,
    heading_z,
    height_01,
    q_from_action,
    shaped_reward,
)

STAGE_STAND = 0
STAGE_VX = 1
STAGE_FULL = 2


def load_train_model(xml: Path | None = None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(xml or TRAIN_XML))
    if int(model.nu) != N_ACT:
        raise RuntimeError(f"H2 nu={model.nu}, expected {N_ACT}")
    pelvis = model.body("pelvis").id
    r_foot = model.body("right_ankle_pitch_link").id
    l_foot = model.body("left_ankle_pitch_link").id
    disable_foot_spheres(model, (r_foot, l_foot))
    disable_mesh_contacts(model)
    cylinders_to_capsules(model)
    _ = pelvis
    return model


def _sample_command(rng: np.random.Generator, stage: int) -> np.ndarray:
    cmd = stand_command()
    cmd[CMD_H] = float(rng.uniform(*HEIGHT_RANGE))
    if rng.random() < REACH_FRAC:
        pitch = float(rng.uniform(0.33, 1.0))
        asym = bool(rng.random() < 0.45)
        t = TeacherIntent(
            r_arm=pitch,
            l_arm=float(rng.uniform(0.20, pitch)) if asym else pitch,
            r_out=float(rng.uniform(-0.2, 0.6)),
            l_out=float(rng.uniform(-0.2, 0.6)),
        )
        cmd[CMD_ARMS] = arm_targets(t)
    elif rng.random() < 0.20:
        cmd[CMD_ARMS] = rng.uniform(-0.4, 0.2, size=14).astype(np.float32)
        cmd[4] = float(rng.uniform(-1.55, 0.0))
        cmd[11] = float(rng.uniform(-1.55, 0.0))
    if (not STAND_ONLY) and stage >= STAGE_VX:
        cmd[CMD_VX] = float(rng.uniform(-0.15, 0.70))
    if (not STAND_ONLY) and stage >= STAGE_FULL:
        cmd[CMD_VY] = float(rng.uniform(-0.15, 0.15))
        cmd[CMD_WZ] = float(rng.uniform(-0.40, 0.40))
    return cmd.astype(np.float32)


class FoundationEnv:
    """One H2, primitive contacts, command-conditioned residual policy."""

    def __init__(self, model: mujoco.MjModel | None = None, *, stage: int = STAGE_STAND):
        self.model = model or load_train_model()
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = self.model.body("pelvis").id
        self.torso_id = self.model.body("torso_link").id
        self.r_foot_id = self.model.body("right_ankle_pitch_link").id
        self.l_foot_id = self.model.body("left_ankle_pitch_link").id
        self.r_geoms = colliding_geoms(self.model, self.r_foot_id)
        self.l_geoms = colliding_geoms(self.model, self.l_foot_id)
        self.r_fg = box_geom(self.model, self.r_geoms, self.r_geoms[0] if self.r_geoms else 0)
        self.l_fg = box_geom(self.model, self.l_geoms, self.l_geoms[0] if self.l_geoms else 0)
        self.qadr, self.vadr = actuator_addrs(self.model)
        self.lo, self.hi = joint_limits(self.model)
        self.kp, self.kd = kp_kd_vectors()
        self.dt = float(self.model.opt.timestep)
        self.stage = int(stage)
        self._rng = np.random.default_rng()
        self.cmd = stand_command()
        self.last_a = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_a = np.zeros(ACT_DIM, dtype=np.float32)
        self._cmd_left = 0
        self._push_left = 0
        self._push_wait = 0
        self._time_left = 0
        self._air_l = 0.0
        self._air_r = 0.0
        self._cmd_frozen = False
        self._pushes = True
        self._horizon = True
        self._off_prev = np.zeros(2, dtype=np.float32)
        self.reset()

    def _tilt(self) -> float:
        return float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])

    def _z(self) -> float:
        return float(self.data.xpos[self.pelvis_id, 2])

    def _hinges(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float32)

    def _qd(self) -> np.ndarray:
        return self.data.qvel[self.vadr].astype(np.float32)

    def _foot_air(self, geoms: list[int]) -> bool:
        if not geoms:
            return False
        z = min(float(self.data.geom_xpos[g, 2]) for g in geoms)
        return z > 0.035

    def _body_vel(self) -> tuple[np.ndarray, float]:
        rot = np.asarray(self.data.xmat[self.torso_id], dtype=np.float64).reshape(3, 3)
        v_w = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        v_b = rot.T @ v_w
        wz = float(self.data.qvel[5])
        return v_b.astype(np.float32), wz

    def _policy_dt(self) -> float:
        return self.dt * DECIMATION

    def _omega(self) -> np.ndarray:
        return np.asarray(self.data.cvel[self.torso_id, :3], dtype=np.float64)

    def _terminated(self) -> bool:
        return self._tilt() < TRAIN_TILT or self._z() < TRAIN_FALL_Z

    def apply_impulse(self, fx: float, fy: float, *, duration_sec: float = PUSH_DUR_SEC) -> None:
        self.data.xfrc_applied[self.torso_id] = 0
        self.data.xfrc_applied[self.torso_id, 0] = float(fx)
        self.data.xfrc_applied[self.torso_id, 1] = float(fy)
        self._push_left = max(1, int(round(float(duration_sec) / self._policy_dt())))

    def _obs(self) -> np.ndarray:
        return build_obs(self.data, self.torso_id, self._hinges(), self._qd(), self.last_a, self.cmd)

    def reset(self, cmd: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0:3] = (0.0, 0.0, SPAWN_Z)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qadr] = STAND_Q
        self.data.qvel[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        nq = 0.03 * self._rng.normal(size=N_ACT)
        self.data.qpos[self.qadr] = np.clip(STAND_Q + nq, self.lo, self.hi)
        self.data.qvel[self.vadr] = 0.04 * self._rng.normal(size=N_ACT)
        if cmd is not None:
            self.data.qpos[self.qadr] = STAND_Q
            self.data.qvel[:] = 0
        self.last_a = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_a = np.zeros(ACT_DIM, dtype=np.float32)
        self.cmd = stand_command() if cmd is None else np.asarray(cmd, dtype=np.float32).reshape(L2_CMD_DIM)
        self._cmd_frozen = cmd is not None
        self._pushes = cmd is None
        self._horizon = cmd is None
        if cmd is None:
            self.cmd = _sample_command(self._rng, self.stage)
        pdt = self._policy_dt()
        self._cmd_left = int(self._rng.integers(100, 201))
        sec = float(self._rng.uniform(*EPISODE_SEC))
        self._time_left = int(round(sec / pdt))
        self._push_left = 0
        self._push_wait = int(round(float(self._rng.uniform(*PUSH_EVERY_SEC)) / pdt))
        self._air_l = 0.0
        self._air_r = 0.0
        self._off_prev = np.zeros(2, dtype=np.float32)
        self.data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _maybe_push(self) -> None:
        if not self._pushes:
            if self._push_left > 0:
                self._push_left -= 1
                if self._push_left <= 0:
                    self.data.xfrc_applied[self.torso_id] = 0
            return
        if self._push_left > 0:
            self._push_left -= 1
            if self._push_left <= 0:
                self.data.xfrc_applied[self.torso_id] = 0
            return
        self._push_wait -= 1
        if self._push_wait > 0:
            self.data.xfrc_applied[self.torso_id] = 0
            return
        mag = float(self._rng.uniform(*PUSH_FORCE))
        ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self.data.xfrc_applied[self.torso_id] = 0
        self.data.xfrc_applied[self.torso_id, 0] = mag * np.cos(ang)
        self.data.xfrc_applied[self.torso_id, 1] = mag * np.sin(ang)
        pdt = self._policy_dt()
        self._push_left = max(1, int(round(PUSH_DUR_SEC / pdt)))
        self._push_wait = int(round(float(self._rng.uniform(*PUSH_EVERY_SEC)) / pdt))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(ACT_DIM), -1.0, 1.0)
        base = q_from_action(self.cmd, a)
        da = a - self.last_a
        dda = a - 2.0 * self.last_a + self.prev_a
        h01 = height_01(float(self.cmd[CMD_H]))
        vx = float(self.cmd[CMD_VX])
        self._maybe_push()
        for _ in range(DECIMATION):
            off = com_err_xy(self.data, self.pelvis_id, self.r_fg, self.l_fg, vx)
            d_off = off - self._off_prev
            self._off_prev = off.copy()
            yaw = heading_z(self.data.qpos)
            q_tgt = np.clip(
                base + balance_delta(body_xy(off, yaw), body_xy(d_off, yaw), height_01=h01, vx=vx),
                self.lo,
                self.hi,
            )
            tau = compute_torques(self.model, self.data, q_tgt, self.kp, self.kd, self.qadr, self.vadr)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        self.prev_a = self.last_a.copy()
        self.last_a = a
        self._cmd_left -= 1
        if self._cmd_left <= 0 and not self._cmd_frozen:
            self.cmd = _sample_command(self._rng, self.stage)
            self._cmd_left = int(self._rng.integers(100, 201))

        v_b, _wz = self._body_vel()
        z = self._z()
        g_z = self._tilt()
        gyro = self._omega()
        v_xy = np.asarray(self.data.qvel[0:2], dtype=np.float64)
        v_cmd = np.array([float(self.cmd[CMD_VX]), float(self.cmd[CMD_VY])], dtype=np.float64)
        p_r = foot_pitch_from_xmat(self.data.xmat[self.r_foot_id])
        p_l = foot_pitch_from_xmat(self.data.xmat[self.l_foot_id])
        q = self._hinges()
        reward = shaped_reward(
            z=z,
            h_cmd=float(self.cmd[CMD_H]),
            tilt=g_z,
            v_b=v_b,
            v_cmd=v_cmd,
            da=da,
            dda=dda,
            gyro=gyro,
            v_xy=v_xy,
            foot_pitch_sq=p_r * p_r + p_l * p_l,
            arm_mse=float(np.mean((q[15:29] - self.cmd[CMD_ARMS]) ** 2)),
        )
        fall = self._terminated()
        timeout = False
        if self._horizon:
            self._time_left -= 1
            timeout = self._time_left <= 0
        if not np.isfinite(z) or not np.isfinite(g_z) or not np.isfinite(reward):
            fall = True
        done = fall or timeout
        if fall:
            reward -= TERMINAL_PENALTY
        reward = float(np.clip(np.nan_to_num(reward, nan=-TERMINAL_PENALTY), -TERMINAL_PENALTY - REWARD_CLIP, REWARD_CLIP))
        return self._obs(), float(reward), bool(done)


class VecFoundationEnv:
    """In-process batch of FoundationEnv. Not multiprocessing — MJX is the GPU path."""

    def __init__(self, n: int, *, stage: int = STAGE_STAND, model: mujoco.MjModel | None = None):
        shared = model or load_train_model()
        self.n = int(n)
        self.envs = [FoundationEnv(shared, stage=stage) for _ in range(self.n)]
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

    @property
    def stage(self) -> int:
        return int(self.envs[0].stage)

    @stage.setter
    def stage(self, value: int) -> None:
        for e in self.envs:
            e.stage = int(value)

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, rew, done = [], [], []
        for i, e in enumerate(self.envs):
            o, r, d = e.step(actions[i])
            if d:
                o = e.reset()
            obs.append(o)
            rew.append(r)
            done.append(d)
        return np.stack(obs), np.asarray(rew, dtype=np.float32), np.asarray(done, dtype=np.float32)


def _policy_act(policy, obs: np.ndarray, device) -> np.ndarray:
    if hasattr(policy, "act_np"):
        return policy.act_np(obs, device=device)
    return np.zeros(ACT_DIM, dtype=np.float32)


def eval_inplace(policy, *, seconds: float = 4.0, device=None) -> dict:
    from agent.l3_eval import eval_suite

    _ = seconds
    return eval_suite(policy, device=device)


def eval_stand(policy, *, seconds: float = 4.0, device=None) -> dict:
    return eval_inplace(policy, seconds=seconds, device=device)


def jax_import_error() -> str | None:
    try:
        import jax  # noqa: F401
    except Exception as exc:
        return f"jax: {exc!r}"
    try:
        from mujoco import mjx  # noqa: F401
    except Exception as exc:
        return f"mjx: {exc!r}"
    return None


def jax_available() -> bool:
    return jax_import_error() is None


def jax_device_kind() -> str:
    try:
        import jax

        dev = jax.devices()[0]
        return f"{dev.platform}:{dev.device_kind}"
    except Exception:
        return "none"


# Trainer aliases (older import names).
jax_import_error = jax_import_error
jax_device_kind = jax_device_kind
VecFoundationEnv = VecFoundationEnv
eval_inplace = eval_inplace
STAGE_STAND = STAGE_STAND
STAGE_VX = STAGE_VX
STAGE_FULL = STAGE_FULL
