"""Native MuJoCo foundation env: 50 Hz policy, 200 Hz Joint-PD, curriculum rewards.

Used for stand eval and as the CPU fallback when MJX-JAX is unavailable.
The MJX twin lives in `agent/l3_mjx.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco

from agent.config import STAND_Z
from agent.h2 import (
    N_ACT,
    SPAWN_Z,
    STAND_Q,
    TRAIN_XML,
    actuator_addrs,
    box_geom,
    colliding_geoms,
    disable_foot_spheres,
    disable_mesh_contacts,
    joint_limits,
)
from agent.joint_pd import compute_torques, kp_kd_vectors
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, CMD_VY, CMD_WZ, L2_CMD_DIM, stand_command
from agent.l3_foundation import (
    ACT_DIM,
    DECIMATION,
    FALL_Z,
    OBS_DIM,
    TILT_LIM,
    balance_delta,
    body_xy,
    build_obs,
    com_err_xy,
    heading_z,
    height_01,
    q_from_action,
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
    _ = pelvis
    return model


def _sample_command(rng: np.random.Generator, stage: int) -> np.ndarray:
    cmd = stand_command()
    cmd[CMD_H] = float(rng.uniform(0.78, 1.02))
    if stage >= STAGE_STAND:
        if rng.random() < 0.35:
            cmd[CMD_ARMS] = rng.uniform(-0.4, 0.2, size=14).astype(np.float32)
            cmd[4] = float(rng.uniform(-1.55, 0.0))
            cmd[11] = float(rng.uniform(-1.55, 0.0))
        if rng.random() < 0.25:
            cmd[CMD_H] = float(rng.uniform(0.70, 0.90))
    if stage >= STAGE_VX:
        cmd[CMD_VX] = float(rng.uniform(-0.15, 0.70))
    if stage >= STAGE_FULL:
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
        self._air_l = 0.0
        self._air_r = 0.0
        self._cmd_frozen = False
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

    def _terminated(self) -> bool:
        return self._tilt() < TILT_LIM or self._z() < FALL_Z

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
        self.last_a = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_a = np.zeros(ACT_DIM, dtype=np.float32)
        self.cmd = stand_command() if cmd is None else np.asarray(cmd, dtype=np.float32).reshape(L2_CMD_DIM)
        self._cmd_frozen = cmd is not None
        if cmd is None:
            self.cmd = _sample_command(self._rng, self.stage)
        self._cmd_left = int(self._rng.integers(150, 250))
        self._push_left = 0
        self._air_l = 0.0
        self._air_r = 0.0
        self._off_prev = np.zeros(2, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _maybe_push(self) -> None:
        if self.stage < STAGE_FULL:
            self.data.xfrc_applied[self.torso_id] = 0
            return
        if self._push_left > 0:
            self._push_left -= 1
            if self._push_left <= 0:
                self.data.xfrc_applied[self.torso_id] = 0
            return
        if self._rng.random() < 0.008:
            f = self._rng.uniform(-60.0, 60.0, size=2)
            self.data.xfrc_applied[self.torso_id, 0:2] = f
            self._push_left = int(self._rng.integers(6, 16))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(ACT_DIM), -1.0, 1.0)
        base = q_from_action(self.cmd, a)
        action_rate = float(np.mean((a - self.prev_a) ** 2))
        tau_acc = 0.0
        h01 = height_01(float(self.cmd[CMD_H]))
        vx = float(self.cmd[CMD_VX])
        for _ in range(DECIMATION):
            self._maybe_push()
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
            tau_acc += float(np.mean(tau * tau))
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        self.prev_a = self.last_a
        self.last_a = a
        self._cmd_left -= 1
        if self._cmd_left <= 0 and not self._cmd_frozen:
            self.cmd = _sample_command(self._rng, self.stage)
            self._cmd_left = int(self._rng.integers(150, 250))

        v_b, wz = self._body_vel()
        z = self._z()
        tilt = self._tilt()
        q = self._hinges()
        r_vx = float(np.exp(-4.0 * (v_b[0] - float(self.cmd[CMD_VX])) ** 2))
        r_vy = float(np.exp(-8.0 * (v_b[1] - float(self.cmd[CMD_VY])) ** 2))
        r_wz = float(np.exp(-6.0 * (wz - float(self.cmd[CMD_WZ])) ** 2))
        r_h = float(np.exp(-12.0 * (z - float(self.cmd[CMD_H])) ** 2))
        r_arm = float(np.exp(-4.0 * np.mean((q[15:29] - self.cmd[CMD_ARMS]) ** 2)))
        r_up = float(np.exp(-6.0 * (1.0 - tilt) ** 2))
        alive = 1.0
        air_l = self._foot_air(self.l_geoms)
        air_r = self._foot_air(self.r_geoms)
        if air_l:
            self._air_l += self.dt * DECIMATION
        else:
            self._air_l = 0.0
        if air_r:
            self._air_r += self.dt * DECIMATION
        else:
            self._air_r = 0.0
        r_air = 0.0
        if self.stage >= STAGE_VX and abs(float(self.cmd[CMD_VX])) > 0.12:
            r_air = float(np.clip(self._air_l + self._air_r, 0.0, 0.4))
            if air_l == air_r:
                r_air -= 0.05
        r_tau = float(np.clip(-1e-6 * tau_acc, -1.5, 0.0))
        r_rate = float(np.clip(-0.15 * action_rate, -1.0, 0.0))
        reward = r_up + r_h + r_arm + alive + r_tau + r_rate
        if self.stage >= STAGE_VX:
            reward += r_vx
        else:
            reward += 0.4 * r_vx
        if self.stage >= STAGE_FULL:
            reward += 0.5 * r_vy + 0.5 * r_wz
        if self.stage >= STAGE_VX:
            reward += r_air
        done = self._terminated()
        if done:
            reward -= 8.0
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


def eval_stand(policy, *, seconds: float = 4.0, device=None) -> dict:
    """Hold idle stand. Pass if tilt/z stay up for the full window."""
    env = FoundationEnv(stage=STAGE_STAND)
    cmd = stand_command()
    obs = env.reset(cmd)
    ticks = int(round(seconds / (env.dt * DECIMATION)))
    z_hist, tilt_hist = [], []
    for _ in range(max(1, ticks)):
        if hasattr(policy, "act_np"):
            a = policy.act_np(obs, device=device)
        else:
            a = np.zeros(ACT_DIM, dtype=np.float32)
        obs, _, done = env.step(a)
        z_hist.append(env._z())
        tilt_hist.append(env._tilt())
        if done:
            break
    held = len(z_hist) >= ticks and min(z_hist) > STAND_Z - 0.08 and min(tilt_hist) > 0.85
    return {
        "ok": bool(held),
        "seconds": float(len(z_hist) * env.dt * DECIMATION),
        "z_min": float(min(z_hist) if z_hist else 0.0),
        "tilt_min": float(min(tilt_hist) if tilt_hist else 0.0),
        "ticks": int(len(z_hist)),
        "need": int(ticks),
    }


def jax_available() -> bool:
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except Exception:
        return False
    return True


def jax_device_kind() -> str:
    if not jax_available():
        return "none"
    import jax

    try:
        dev = jax.devices()[0]
        return f"{dev.platform}:{dev.device_kind}"
    except Exception:
        return "jax-unknown"
