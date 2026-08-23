"""MJX-JAX batched H2 env (ROCm-capable). Primitive collisions, 50 Hz / 200 Hz PD.

Falls back to raising ImportError if `jax` or `mujoco.mjx` is missing — the trainer
then uses `agent.l3_env.VecFoundationEnv`. Mesh contacts are already off on the
training MjModel before `mjx.put_model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.h2 import (
    L_AP,
    L_HP,
    L_HR,
    R_AP,
    R_HP,
    R_HR,
    SPAWN_Z,
    STAND_COM_X,
    STAND_Q,
    SQUAT_Q,
    actuator_addrs,
    box_geom,
    colliding_geoms,
    joint_limits,
)
from agent.joint_pd import kp_kd_vectors
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, CMD_VY, CMD_WZ, L2_CMD_DIM
from agent.l3_env import STAGE_FULL, STAGE_STAND, STAGE_VX, load_train_model
from agent.l3_foundation import ACT_DIM, ACTION_SCALE, DECIMATION, FALL_Z, HIDDEN, OBS_DIM, TILT_LIM


def _require_jax():
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    return jax, jp, mjx


def mlp_init(jax, jp, key, *, zero_out: bool = True):
    sizes = (OBS_DIM, *HIDDEN, ACT_DIM)
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        if zero_out and i == len(sizes) - 2:
            w = jp.zeros((n_in, n_out), dtype=jp.float32)
        else:
            w = jax.random.normal(keys[i], (n_in, n_out), dtype=jp.float32) * jp.sqrt(2.0 / n_in)
        b = jp.zeros((n_out,), dtype=jp.float32)
        params.append((w, b))
    return params


def mlp_forward(jp, params, obs):
    x = obs
    for w, b in params[:-1]:
        x = jax_silu(jp, x @ w + b)
    w, b = params[-1]
    return x @ w + b


def jax_silu(jp, x):
    return x * jp.sigmoid(x)


@dataclass
class MjxSpec:
    qadr: np.ndarray
    vadr: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_lo: np.ndarray
    tau_hi: np.ndarray
    pelvis_id: int
    torso_id: int
    r_foot_id: int
    l_foot_id: int
    r_geoms: tuple[int, ...]
    l_geoms: tuple[int, ...]
    r_fg: int
    l_fg: int
    nq: int
    nv: int
    nu: int


def make_spec(model) -> MjxSpec:
    qadr, vadr = actuator_addrs(model)
    lo, hi = joint_limits(model)
    kp, kd = kp_kd_vectors()
    r_foot = model.body("right_ankle_pitch_link").id
    l_foot = model.body("left_ankle_pitch_link").id
    return MjxSpec(
        qadr=qadr,
        vadr=vadr,
        lo=lo,
        hi=hi,
        kp=kp,
        kd=kd,
        tau_lo=model.actuator_ctrlrange[:, 0].astype(np.float32),
        tau_hi=model.actuator_ctrlrange[:, 1].astype(np.float32),
        pelvis_id=int(model.body("pelvis").id),
        torso_id=int(model.body("torso_link").id),
        r_foot_id=int(r_foot),
        l_foot_id=int(l_foot),
        r_geoms=tuple(colliding_geoms(model, r_foot)),
        l_geoms=tuple(colliding_geoms(model, l_foot)),
        r_fg=box_geom(model, colliding_geoms(model, r_foot), colliding_geoms(model, r_foot)[0] if colliding_geoms(model, r_foot) else 0),
        l_fg=box_geom(model, colliding_geoms(model, l_foot), colliding_geoms(model, l_foot)[0] if colliding_geoms(model, l_foot) else 0),
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
    )


def try_put_model(model):
    _, _, mjx = _require_jax()
    return mjx.put_model(model)


class MjxFoundationEnv:
    """Batched MJX env. `reset`/`step` take and return JAX arrays with leading env axis."""

    def __init__(self, n: int, *, stage: int = STAGE_STAND):
        jax, jp, mjx = _require_jax()
        self.jax, self.jp, self.mjx = jax, jp, mjx
        self.n = int(n)
        self.stage = int(stage)
        self.model = load_train_model()
        self.spec = make_spec(self.model)
        self.mx = try_put_model(self.model)
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        self._constants()
        self._compile()

    def _constants(self) -> None:
        jp = self.jp
        s = self.spec
        self.qadr = jp.asarray(s.qadr)
        self.vadr = jp.asarray(s.vadr)
        self.lo = jp.asarray(s.lo)
        self.hi = jp.asarray(s.hi)
        self.kp = jp.asarray(s.kp)
        self.kd = jp.asarray(s.kd)
        self.tau_lo = jp.asarray(s.tau_lo)
        self.tau_hi = jp.asarray(s.tau_hi)
        self.stand_q = jp.asarray(STAND_Q)
        self.squat_q = jp.asarray(SQUAT_Q)
        self.upper = jp.asarray(np.arange(15, 29, dtype=np.int32))

    def _xmat3(self, dx, body: int):
        jp = self.jp
        mat = dx.xmat[body]
        return jp.reshape(mat, (3, 3))

    def _sample_cmd(self, rng):
        jax, jp = self.jax, self.jp
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
        h = jax.random.uniform(k1, (), minval=0.78, maxval=1.02)
        cmd = jp.zeros((L2_CMD_DIM,), dtype=jp.float32).at[CMD_H].set(h)
        arms_on = jax.random.bernoulli(k2, 0.35)
        arms = jax.random.uniform(k3, (14,), minval=-0.4, maxval=0.2)
        sp = jax.random.uniform(k4, (2,), minval=-1.55, maxval=0.0)
        arms = arms.at[0].set(sp[0]).at[7].set(sp[1])
        cmd = jp.where(arms_on, cmd.at[4:18].set(arms), cmd)
        squat = jax.random.bernoulli(k5, 0.25)
        cmd = jp.where(squat, cmd.at[CMD_H].set(jax.random.uniform(k6, (), minval=0.70, maxval=0.90)), cmd)
        if self.stage >= STAGE_VX:
            rng, k = jax.random.split(rng)
            cmd = cmd.at[CMD_VX].set(jax.random.uniform(k, (), minval=-0.15, maxval=0.70))
        if self.stage >= STAGE_FULL:
            rng, k1, k2 = jax.random.split(rng, 3)
            cmd = cmd.at[CMD_VY].set(jax.random.uniform(k1, (), minval=-0.15, maxval=0.15))
            cmd = cmd.at[CMD_WZ].set(jax.random.uniform(k2, (), minval=-0.40, maxval=0.40))
        return rng, cmd

    def _default_q(self, cmd):
        jp = self.jp
        h = jp.clip((cmd[CMD_H] - 0.62) / 0.40, 0.0, 1.0)
        q = (1.0 - h) * self.squat_q + h * self.stand_q
        q = q.at[self.upper].set(cmd[4:18])
        return q

    def _balance(self, dx, cmd, off_prev):
        jp = self.jp
        com = dx.subtree_com[self.spec.pelvis_id, :2]
        feet = 0.5 * (dx.geom_xpos[self.spec.r_fg, :2] + dx.geom_xpos[self.spec.l_fg, :2])
        vx = cmd[CMD_VX]
        off = com - feet - jp.array([STAND_COM_X + 0.035 * vx, 0.0])
        dlt_w = off - off_prev
        w, x, y, z = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
        yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = jp.cos(yaw), jp.sin(yaw)
        err_x = c * off[0] + s * off[1]
        err_y = -s * off[0] + c * off[1]
        d_x = c * dlt_w[0] + s * dlt_w[1]
        d_y = -s * dlt_w[0] + c * dlt_w[1]
        h01 = jp.clip((cmd[CMD_H] - 0.62) / 0.40, 0.0, 1.0)
        scale = jp.clip(h01, 0.5, 1.0)
        scale = jp.where(jp.abs(vx) > 0.08, scale * jp.clip(1.0 - (jp.abs(vx) - 0.08) / 0.42, 0.2, 1.0), scale)
        ak_lim = jp.clip(0.14 + 0.8 * jp.abs(err_x), 0.14, 0.28)
        hy_lim = jp.clip(0.20 + 1.5 * jp.abs(err_x), 0.20, 0.50)
        sag_ak = jp.clip(2.4 * err_x - 5.0 * d_x, -ak_lim, ak_lim) * scale
        sag_hy = jp.clip(4.0 * err_x - 8.0 * d_x, -hy_lim, hy_lim) * scale
        pd_hx = jp.clip(1.2 * err_y - 6.0 * d_y, -0.25, 0.25)
        dlt = jp.zeros((ACT_DIM,), dtype=jp.float32)
        dlt = dlt.at[R_AP].set(sag_ak).at[L_AP].set(sag_ak)
        dlt = dlt.at[R_HP].set(sag_hy).at[L_HP].set(sag_hy)
        dlt = dlt.at[R_HR].set(pd_hx).at[L_HR].set(pd_hx)
        return dlt, off

    def _pd(self, dx, q_tgt):
        jp = self.jp
        q = dx.qpos[self.qadr]
        qd = dx.qvel[self.vadr]
        tau = self.kp * (q_tgt - q) - self.kd * qd + dx.qfrc_bias[self.vadr]
        return jp.clip(tau, self.tau_lo, self.tau_hi)

    def _obs(self, dx, last_a, cmd):
        jp = self.jp
        rot = self._xmat3(dx, self.spec.torso_id)
        grav = rot.T @ jp.array([0.0, 0.0, -1.0])
        gyro = dx.cvel[self.spec.torso_id, :3]
        q = dx.qpos[self.qadr]
        qd = dx.qvel[self.vadr]
        return jp.concatenate([grav, gyro, q - self.stand_q, qd, last_a, cmd]).astype(jp.float32)

    def _reset_one(self, rng):
        jax, jp, mjx = self.jax, self.jp, self.mjx
        dx = mjx.make_data(self.mx)
        qpos = dx.qpos.at[0:3].set(jp.array([0.0, 0.0, SPAWN_Z])).at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qpos = qpos.at[self.qadr].set(self.stand_q)
        dx = dx.replace(qpos=qpos, qvel=jp.zeros_like(dx.qvel))
        dx = mjx.forward(self.mx, dx)
        rng, cmd = self._sample_cmd(rng)
        last_a = jp.zeros((ACT_DIM,), dtype=jp.float32)
        off_prev = jp.zeros((2,), dtype=jp.float32)
        return rng, dx, last_a, cmd, last_a, jp.array(200, dtype=jp.int32), off_prev

    def _step_one(self, rng, dx, last_a, cmd, prev_a, cmd_left, off_prev, action):
        jax, jp, mjx = self.jax, self.jp, self.mjx
        a = jp.clip(action, -1.0, 1.0)
        base = self._default_q(cmd) + ACTION_SCALE * a
        tau_acc = 0.0

        def body(i, carry):
            dx_i, acc, off_i = carry
            dlt, off_i = self._balance(dx_i, cmd, off_i)
            q_tgt = jp.clip(base + dlt, self.lo, self.hi)
            tau = self._pd(dx_i, q_tgt)
            dx_i = dx_i.replace(ctrl=tau)
            dx_i = mjx.step(self.mx, dx_i)
            return dx_i, acc + jp.mean(tau * tau), off_i

        dx, tau_acc, off_prev = jax.lax.fori_loop(0, DECIMATION, body, (dx, jp.float32(0.0), off_prev))
        rot = self._xmat3(dx, self.spec.torso_id)
        v_b = rot.T @ dx.qvel[0:3]
        wz = dx.qvel[5]
        z = dx.xpos[self.spec.pelvis_id, 2]
        tilt = rot[2, 2]
        q = dx.qpos[self.qadr]
        r_vx = jp.exp(-4.0 * (v_b[0] - cmd[CMD_VX]) ** 2)
        r_vy = jp.exp(-8.0 * (v_b[1] - cmd[CMD_VY]) ** 2)
        r_wz = jp.exp(-6.0 * (wz - cmd[CMD_WZ]) ** 2)
        r_h = jp.exp(-12.0 * (z - cmd[CMD_H]) ** 2)
        r_arm = jp.exp(-4.0 * jp.mean((q[15:29] - cmd[CMD_ARMS]) ** 2))
        r_up = jp.exp(-6.0 * (1.0 - tilt) ** 2)
        r_rate = jp.clip(-0.15 * jp.mean((a - prev_a) ** 2), -1.0, 0.0)
        r_tau = jp.clip(-1e-6 * tau_acc, -1.5, 0.0)
        reward = r_up + r_h + r_arm + 1.0 + r_tau + r_rate + (r_vx if self.stage >= STAGE_VX else 0.4 * r_vx)
        if self.stage >= STAGE_FULL:
            reward = reward + 0.5 * r_vy + 0.5 * r_wz
        if self.stage >= STAGE_VX:
            r_geoms = self.spec.r_geoms[:1]
            l_geoms = self.spec.l_geoms[:1]
            air_r = dx.geom_xpos[r_geoms[0], 2] > 0.035 if r_geoms else jp.bool_(False)
            air_l = dx.geom_xpos[l_geoms[0], 2] > 0.035 if l_geoms else jp.bool_(False)
            r_air = jp.where(air_l != air_r, jp.float32(0.05), jp.float32(-0.05))
            reward = reward + jp.where(jp.abs(cmd[CMD_VX]) > 0.12, r_air, 0.0)
        done = (tilt < TILT_LIM) | (z < FALL_Z)
        reward = reward - jp.where(done, 8.0, 0.0)
        cmd_left = cmd_left - 1
        rng, new_cmd = self._sample_cmd(rng)
        cmd = jp.where(cmd_left <= 0, new_cmd, cmd)
        cmd_left = jp.where(cmd_left <= 0, jp.array(200, dtype=jp.int32), cmd_left)
        rng, dx_r, last_a_r, cmd_r, prev_r, left_r, off_r = self._reset_one(rng)
        dx = jax.tree_util.tree_map(lambda a, b: jp.where(done, b, a), dx, dx_r)
        last_a_out = jp.where(done, last_a_r, a)
        cmd = jp.where(done, cmd_r, cmd)
        prev_a = jp.where(done, prev_r, last_a)
        cmd_left = jp.where(done, left_r, cmd_left)
        off_prev = jp.where(done, off_r, off_prev)
        obs = self._obs(dx, last_a_out, cmd)
        return rng, dx, last_a_out, cmd, prev_a, cmd_left, off_prev, obs, reward, done

    def _compile(self) -> None:
        jax = self.jax
        self._reset_v = jax.jit(jax.vmap(self._reset_one))
        self._step_v = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))

    def reset(self, rng):
        rngs = self.jax.random.split(rng, self.n)
        rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev = self._reset_v(rngs)
        obs = jax.vmap(self._obs)(dx, last_a, cmd)
        self._state = (rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev)
        return obs

    def step(self, actions):
        rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev = self._state
        rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev, obs, rew, done = self._step_v(
            rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev, actions
        )
        self._state = (rngs, dx, last_a, cmd, prev_a, cmd_left, off_prev)
        return obs, rew, done
