"""Command-conditioned whole-body foundation policy (Level 3).

Obs (119): grav(3)+gyro(3)+qerr(31)+qvel(31)+last_a(31)+cmd(18)+unused phase slots(2).
last_a is a_{t-1} (31-DoF previous motor command) so ankles can damp phase lag.
Act (31): residual around a command-conditioned default pose.
Policy rate 50 Hz; Joint-PD at 200 Hz (decimation=4).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agent.config import STAND_Z
from agent.h2 import (
    L_AP,
    L_HP,
    L_HR,
    N_ACT,
    R_AP,
    R_HP,
    R_HR,
    STAND_COM_X,
    STAND_Q,
    SQUAT_Q,
)
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, L2_CMD_DIM, UPPER_IDX

DECIMATION = 10  # 0.002 s physics → 50 Hz policy
ACTION_SCALE = 0.5
# Low-pass stochastic PPO targets. Independent 50 Hz noise was exciting every
# H2 joint and causing falls unrelated to useful exploration.
ACTION_FILTER_ALPHA = 0.10
LEGACY_OBS_DIM = 3 + 3 + N_ACT + N_ACT + N_ACT + L2_CMD_DIM  # 117
OBS_DIM = LEGACY_OBS_DIM + 2  # sin(phi), cos(phi)
ACT_DIM = N_ACT
TILT_LIM = 0.65  # app / engine
FALL_Z = 0.40
# Stage A: ~21°. Walk: ~29°. 0.83 (~34°) let a falling log collect vx reward for ~2s.
TRAIN_TILT = 0.87
TRAIN_FALL_Z = 0.40
# Forward speed counts only while the torso is still upright. Idle stand sits at
# tilt≈0.996; the falling log is below 0.97 by the time it is actually fast.
VEL_TILT = 0.92
VEL_UPRIGHT = 0.97
VEL_UPRIGHT_SPAN = 0.026
# Body-frame z floors. Deep squat at h=0.65 still has knee≈0.21; kneeling is ~0.08.
CONTACT_Z_KNEE = 0.14
CONTACT_Z_HAND = 0.12
CONTACT_Z_ELBOW = 0.15
TERMINAL_PENALTY = 150.0
REWARD_CLIP = 12.0
EPISODE_SEC = (15.0, 20.0)
HEIGHT_RANGE = (0.48, 0.78)
REACH_FRAC = 0.0
# Of reach samples: left-only / right-only / both.
ARM_LEFT_FRAC = 0.30
ARM_RIGHT_FRAC = 0.30
SQUAT_FRAC = 0.0
SQUAT_H_STAND = 0.78
SQUAT_H_LOW = 0.50
SQUAT_DOWN_SEC = 1.5
SQUAT_HOLD_SEC = 5.0
SQUAT_UP_SEC = 1.5
POLICY_DT = 0.002 * DECIMATION  # matches scene_train timestep
SQUAT_TICKS = int(round((SQUAT_DOWN_SEC + SQUAT_HOLD_SEC + SQUAT_UP_SEC) / POLICY_DT))
PUSH_EVERY_SEC = (2.0, 3.0)
PUSH_DUR_SEC = 0.20
# Toward the 50 N eval; still below the eval impulse so Stage A can survive.
PUSH_FORCE = (12.0, 22.0)
ALIVE_BONUS = 1.0
ANG_VEL_COEF = 0.50
ANG_VEL_CLIP = 4.0
LIN_VEL_COEF = 0.50
RATE_COEF = 0.04
QVEL_COEF = 0.002
QVEL_CLIP = 2.0
AIR_COEF = 0.80
# Landing bonus after a real swing. No phase clock: either foot may step.
AIR_TIME_MIN = 0.12
AIR_TIME_SCALE = 2.0
AIR_TIME_CAP = 0.8
# A crawl may slide the feet. 1.5 cost more than the first centimeters of speed.
SLIP_COEF = 0.4
# +2 per step at the commanded speed. The old exp kernel paid ~0.3 there and ~0
# at a stand, so polishing the stand beat walking.
VEL_REWARD_SCALE = 2.0
# Scale of the hardcoded IPM prior. 1.0 was a limit-cycle; policy must learn the rest.
BALANCE_PRIOR_SCALE = 0.25
STAND_ONLY = False
WALK_ONLY = True
# Phase-aware crawl first. Idle stand actor survives <=0.16 m/s for 10 s,
# while >=0.20 m/s falls before learning; raise this only after crawl eval passes.
VX_RANGE = (0.09, 0.11)
VX_ZERO_FRAC = 0.35
HIDDEN = (256, 256, 128)


class HumanoidFoundationPolicy(nn.Module):
    def __init__(self, *, zero_out: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        prev = OBS_DIM
        for h in HIDDEN:
            layers.extend([nn.Linear(prev, h), nn.SiLU()])
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.net = nn.Sequential(*layers)
        if zero_out:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accept legacy 117-D actors; phase columns start at zero influence."""
        state_dict = upgrade_actor_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Action in [-1, 1]. q_from_action applies ACTION_SCALE."""
        return torch.tanh(self.forward(obs))

    def act_np(self, obs: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device or next(self.parameters()).device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        with torch.no_grad():
            a = self.act(x).detach().cpu().numpy().astype(np.float32)
        return a[0] if squeeze else a


def linear_indices() -> tuple[int, ...]:
    """Indices of nn.Linear modules inside HumanoidFoundationPolicy.net."""
    return (0, 2, 4, 6)


def upgrade_actor_state_dict(sd: dict) -> dict:
    """Pad a legacy actor's first layer from obs 117 to 119 without changing outputs."""
    key = "net.0.weight"
    w = sd.get(key)
    if not torch.is_tensor(w) or tuple(w.shape) != (HIDDEN[0], LEGACY_OBS_DIM):
        return sd
    out = dict(sd)
    padded = torch.zeros((HIDDEN[0], OBS_DIM), dtype=w.dtype, device=w.device)
    padded[:, :LEGACY_OBS_DIM] = w
    out[key] = padded
    return out


def jax_params_to_state_dict(params) -> dict:
    """JAX kernels are (in, out); PyTorch Linear is (out, in)."""
    sd: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(linear_indices()):
        w, b = params[i]
        sd[f"net.{layer}.weight"] = torch.as_tensor(np.array(w, copy=True).T)
        sd[f"net.{layer}.bias"] = torch.as_tensor(np.array(b, copy=True))
    return sd


def state_dict_to_jax(sd: dict):
    import jax.numpy as jp

    sd = upgrade_actor_state_dict(sd)
    params = []
    for layer in linear_indices():
        w = np.asarray(sd[f"net.{layer}.weight"].detach().cpu().numpy(), dtype=np.float32).T
        b = np.asarray(sd[f"net.{layer}.bias"].detach().cpu().numpy(), dtype=np.float32)
        params.append((jp.asarray(w), jp.asarray(b)))
    return params


def torso_imu(data, torso_id: int) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    grav = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gyro = np.asarray(data.cvel[torso_id, :3], dtype=np.float64)
    return grav.astype(np.float32), gyro.astype(np.float32)


def squat_cmd_height(t_sec: float) -> float:
    """Stand → 0.70 m → stand. Matches eval_deep_squat / UI «присядь»."""
    t = float(t_sec)
    down, hold, up = float(SQUAT_DOWN_SEC), float(SQUAT_HOLD_SEC), float(SQUAT_UP_SEC)
    h0, h1 = float(SQUAT_H_STAND), float(SQUAT_H_LOW)
    if t < down:
        return h0 + (h1 - h0) * (t / max(down, 1e-6))
    if t < down + hold:
        return h1
    if t < down + hold + up:
        return h1 + (h0 - h1) * ((t - down - hold) / max(up, 1e-6))
    return h0


def height_01(h_m: float) -> float:
    return float(np.clip((float(h_m) - 0.48) / max(STAND_Z - 0.48, 1e-3), 0.0, 1.0))


def apply_walk_gait(q: np.ndarray, vx: float, phi: float) -> np.ndarray:
    """Stand pose only. The step has to come from the policy, not a joint clock."""
    del vx, phi
    return q


def advance_gait_phi(phi: float, vx: float, dt: float) -> float:
    """No gait phase. Obs keeps the two phase slots so older actors still load."""
    del phi, vx, dt
    return 0.0


def foot_air_bonus(air_t_l: float, air_t_r: float, air_l: bool, air_r: bool, moving: bool) -> tuple[float, float, float]:
    """Reward a foot for having been up, paid once when it lands. Returns new timers."""
    bonus = 0.0
    if moving:
        if air_t_l > 0.0 and not air_l:
            bonus += AIR_TIME_SCALE * min(AIR_TIME_CAP, max(0.0, float(air_t_l) - AIR_TIME_MIN))
        if air_t_r > 0.0 and not air_r:
            bonus += AIR_TIME_SCALE * min(AIR_TIME_CAP, max(0.0, float(air_t_r) - AIR_TIME_MIN))
    t_l = float(air_t_l) + POLICY_DT if air_l else 0.0
    t_r = float(air_t_r) + POLICY_DT if air_r else 0.0
    return float(bonus), t_l, t_r


def default_q(cmd: np.ndarray, phi: float = 0.0) -> np.ndarray:
    """Stand/squat lerp + arms. Residual policy supplies locomotion."""
    cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
    h = height_01(float(cmd[CMD_H]) if cmd.shape[0] > CMD_H else STAND_Z)
    q = ((1.0 - h) * SQUAT_Q + h * STAND_Q).astype(np.float32)
    if cmd.shape[0] >= L2_CMD_DIM:
        q[list(UPPER_IDX)] = cmd[CMD_ARMS]
    vx = float(cmd[CMD_VX]) if cmd.shape[0] > CMD_VX else 0.0
    return apply_walk_gait(q, vx, phi)


def q_from_action(cmd: np.ndarray, action: np.ndarray, phi: float = 0.0) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(N_ACT), -1.0, 1.0)
    return default_q(cmd, phi) + ACTION_SCALE * a


def heading_z(qpos: np.ndarray) -> float:
    w, x, y, z = (float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6]))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def com_err_xy(
    data,
    pelvis_id: int,
    r_fg: int,
    l_fg: int,
    vx: float = 0.0,
    gait_phi: float = 0.0,
) -> np.ndarray:
    """World-frame COM error relative to the midpoint of the feet."""
    del vx, gait_phi
    com = np.asarray(data.subtree_com[pelvis_id, :2], dtype=np.float32)
    right = np.asarray(data.geom_xpos[r_fg, :2], dtype=np.float32)
    left = np.asarray(data.geom_xpos[l_fg, :2], dtype=np.float32)
    support = 0.5 * (right + left)
    return com - support - np.array([STAND_COM_X, 0.0], dtype=np.float32)


def body_xy(world_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    x, y = float(world_xy[0]), float(world_xy[1])
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


def balance_delta(err_xy: np.ndarray, d_xy: np.ndarray, *, height_01: float = 1.0, vx: float = 0.0) -> np.ndarray:
    """Ankle/hip inverted-pendulum prior on top of default_q. Not a gait.

    Joint-PD tracks a pose; this pose must include CoM feedback or a 70 kg H2
    folds. Zero policy residual still stands. Same signs as the old stance PD.
    """
    err_x, err_y = float(err_xy[0]), float(err_xy[1])
    d_x, d_y = float(d_xy[0]), float(d_xy[1])
    scale = float(np.clip(height_01, 0.5, 1.0))
    ak_lim = float(np.clip(0.14 + 0.8 * abs(err_x), 0.14, 0.28))
    hy_lim = float(np.clip(0.20 + 1.5 * abs(err_x), 0.20, 0.50))
    sag_ak = float(np.clip(2.4 * err_x - 5.0 * d_x, -ak_lim, ak_lim)) * scale
    sag_hy = float(np.clip(4.0 * err_x - 8.0 * d_x, -hy_lim, hy_lim)) * scale
    pd_hx = float(np.clip(1.2 * err_y - 6.0 * d_y, -0.25, 0.25))
    dlt = np.zeros(N_ACT, dtype=np.float32)
    dlt[R_AP] = dlt[L_AP] = sag_ak
    dlt[R_HP] = dlt[L_HP] = sag_hy
    dlt[R_HR] = dlt[L_HR] = pd_hx
    return dlt * float(BALANCE_PRIOR_SCALE)


def moving_velocity_reward(v_b, v_cmd, tilt: float) -> float:
    """0 while standing, +VEL_REWARD_SCALE at the command, nothing extra for falling faster."""
    vel = np.asarray(v_b[:2], dtype=np.float64)
    cmd = np.asarray(v_cmd, dtype=np.float64)
    speed = float(np.linalg.norm(cmd))
    along = float(np.dot(vel, cmd) / max(speed, 1e-6))
    progress = float(np.clip(along, 0.0, speed) / max(speed, 1e-6))
    upright = float(np.clip((float(tilt) - VEL_UPRIGHT) / VEL_UPRIGHT_SPAN, 0.0, 1.0))
    over = float(np.clip(along - speed - 0.15, 0.0, 2.0))
    return float(np.clip(VEL_REWARD_SCALE * progress * upright - over, -2.0, 2.0))


def shaped_reward(
    *,
    z: float,
    h_cmd: float,
    tilt: float,
    v_b: np.ndarray,
    v_cmd: np.ndarray,
    da: np.ndarray,
    dda: np.ndarray,
    gyro: np.ndarray,
    v_xy: np.ndarray,
    foot_pitch_sq: float,
    arm_mse: float,
    qvel: np.ndarray | None = None,
    air_l: bool = False,
    air_r: bool = False,
    slip_foot: float = 0.0,
    air_bonus: float = 0.0,
) -> float:
    """Dense stand-first reward. Terminal fall penalty is applied by the env."""
    r_alive = float(ALIVE_BONUS)
    r_h = float(np.exp(-10.0 * abs(float(z) - float(h_cmd))))
    r_up = float(np.exp(-5.0 * (1.0 - float(tilt) * float(tilt))))
    dv = np.asarray(v_b[:2], dtype=np.float64) - np.asarray(v_cmd, dtype=np.float64)
    vnorm = float(np.linalg.norm(v_cmd))
    vel_k = 8.0 if vnorm >= 0.08 else 2.0
    if vnorm >= 0.08:
        r_vel = moving_velocity_reward(v_b, v_cmd, tilt)
    else:
        r_vel = float(np.exp(-vel_k * float(np.sum(dv ** 2))))
    r_rate = float(np.clip(-RATE_COEF * float(np.sum(np.asarray(da, dtype=np.float64) ** 2)), -1.0, 0.0))
    r_acc = float(np.clip(-0.005 * float(np.sum(np.asarray(dda, dtype=np.float64) ** 2)), -1.0, 0.0))
    gx, gy = float(gyro[0]), float(gyro[1])
    r_ang = float(np.clip(-ANG_VEL_COEF * (gx * gx + gy * gy), -ANG_VEL_CLIP, 0.0))
    r_lin = 0.0
    if vnorm < 0.05:
        r_lin = float(np.clip(-LIN_VEL_COEF * float(np.sum(np.asarray(v_xy, dtype=np.float64) ** 2)), -2.0, 0.0))
    r_foot = float(np.clip(-0.1 * float(foot_pitch_sq), -1.0, 0.0))
    r_arm = 0.4 * float(np.exp(-4.0 * float(arm_mse)))
    r_qvel = 0.0
    if qvel is not None and vnorm < 0.08:
        r_qvel = float(np.clip(-QVEL_COEF * float(np.sum(np.asarray(qvel, dtype=np.float64) ** 2)), -QVEL_CLIP, 0.0))
    r_air = float(air_bonus)
    r_slip = 0.0
    if vnorm >= 0.08:
        if air_l and air_r:
            r_air -= float(AIR_COEF)
        r_slip = float(np.clip(-SLIP_COEF * float(slip_foot), -2.0, 0.0))
    return float(
        r_alive + r_h + r_up + r_vel + r_rate + r_acc + r_ang + r_lin + r_foot + r_arm + r_qvel + r_air + r_slip
    )


def foot_pitch_from_xmat(xmat) -> float:
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(-R[2, 0], R[2, 2]))


def foot_world_xy_speed_sq(xmat, cvel) -> float:
    """World-frame horizontal speed² of a body COM. cvel lin is body-frame."""
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    v = R @ np.asarray(cvel, dtype=np.float64).reshape(-1)[3:6]
    return float(v[0] * v[0] + v[1] * v[1])


def build_obs(
    data,
    torso_id: int,
    q: np.ndarray,
    qd: np.ndarray,
    last_a: np.ndarray,
    cmd: np.ndarray,
    gait_phi: float = 0.0,
) -> np.ndarray:
    grav, gyro = torso_imu(data, torso_id)
    q = np.asarray(q, dtype=np.float32).reshape(N_ACT)
    qd = np.asarray(qd, dtype=np.float32).reshape(N_ACT)
    last_a = np.asarray(last_a, dtype=np.float32).reshape(N_ACT)
    cmd = np.asarray(cmd, dtype=np.float32).reshape(L2_CMD_DIM)
    phase = np.array([np.sin(gait_phi), np.cos(gait_phi)], dtype=np.float32)
    return np.concatenate([grav, gyro, q - STAND_Q, qd, last_a, cmd, phase]).astype(np.float32)
