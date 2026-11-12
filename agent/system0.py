"""System 0 (Spinal Cord): Free-standing balance, RL locomotion, Torque-PD & 6D Whole-Body IK."""

from __future__ import annotations

import numpy as np
import mujoco

from agent.config import DECIMATION, FALL_Z, PHYSICS_DT, SLEEP_ACTION_DIM, SLEEP_PROPRIO_DIM, SLEW_RAD, TILT_LIM
from agent.dex3 import FINGER_KD, FINGER_KP, HOOK_LOCAL_L, HOOK_LOCAL_R, finger_pose
from agent.flip_policy import FlipPolicy, flip_torque, load_g1_flip
from agent.g1_walk import G1LegPolicy, load_g1_walk
from agent.standup import G1StandUpPolicy, load_g1_standup
from agent.h2 import (
    KD,
    KP,
    L_AP,
    L_HP,
    L_HR,
    N_ACT,
    R_AP,
    R_HP,
    R_HR,
    SPAWN_XY,
    SPAWN_Z,
    SQUAT_Q,
    STAND_COM_X,
    STAND_Q,
    WAIST_P,
    WAIST_R,
    WAIST_Y,
    arm_hang_cmd,
)
from agent.joint_pd import compute_torques

# Forward-and-slightly-down wrist orientation for horizontal handles / buttons (oven, drawer, toaster)
MAT_HORIZ_HOOK = np.array([
    [0.80, -0.60, 0.0],
    [0.0, 0.0, 1.0],
    [-0.60, -0.80, 0.0],
], dtype=np.float64)

# Forward-and-sideways wrist orientation for vertical handles (fridge, cabinet, doors)
MAT_VERT_HOOK = np.array([
    [0.80, -0.60, 0.0],
    [0.60, 0.80, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# Forward palm orientation for front pushes / buttons
MAT_PUSH_FRONT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# 20-D Raw Action bounds:
# [vx(1), wz(1), squat(1), waist_yaw(1), waist_pitch(1), right_arm(7), left_arm(7), right_grip(1)]
ACT_LO = np.array([
    -0.45, -0.85, 0.0, -0.85, -0.25,
    -2.20, -1.50, -1.60, -0.45, -1.95, -1.55, -1.60,
    -2.20, -0.60, -1.60, -0.45, -1.95, -1.55, -1.60,
    0.0,
], dtype=np.float32)

ACT_HI = np.array([
    0.70, 0.85, 0.75, 0.85, 0.30,
    1.20, 0.60, 1.60, 1.95, 1.95, 1.55, 1.60,
    1.20, 1.50, 1.60, 1.95, 1.95, 1.55, 1.60,
    1.0,
], dtype=np.float32)

ACT_MID = 0.5 * (ACT_LO + ACT_HI)
ACT_SPAN = np.maximum(0.5 * (ACT_HI - ACT_LO), 1e-4)


def default_raw_action() -> np.ndarray:
    """Standing neutral 20-D raw action vector."""
    a = np.zeros(SLEEP_ACTION_DIM, dtype=np.float32)
    a[5:12] = STAND_Q[22:29]
    a[12:19] = STAND_Q[15:22]
    return a


def raw_to_norm_action(raw: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(raw, dtype=np.float32).reshape(SLEEP_ACTION_DIM), ACT_LO, ACT_HI)
    return np.clip((r - ACT_MID) / ACT_SPAN, -1.0, 1.0).astype(np.float32)


def norm_to_raw_action(norm: np.ndarray) -> np.ndarray:
    n = np.clip(np.asarray(norm, dtype=np.float32).reshape(SLEEP_ACTION_DIM), -1.0, 1.0)
    return np.clip(ACT_MID + n * ACT_SPAN, ACT_LO, ACT_HI).astype(np.float32)


def heading_z(qpos_quat: np.ndarray) -> float:
    w, x, y, z = (float(qpos_quat[0]), float(qpos_quat[1]), float(qpos_quat[2]), float(qpos_quat[3]))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def body_xy(world_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    x, y = float(world_xy[0]), float(world_xy[1])
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


def torso_imu(data: mujoco.MjData, torso_id: int) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    grav = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gyro = np.asarray(data.cvel[torso_id, :3], dtype=np.float64)
    return grav.astype(np.float32), gyro.astype(np.float32)


def com_err_xy(data: mujoco.MjData, pelvis_id: int, r_fg: int, l_fg: int) -> np.ndarray:
    com = np.asarray(data.subtree_com[pelvis_id, :2], dtype=np.float32)
    right = np.asarray(data.geom_xpos[r_fg, :2], dtype=np.float32)
    left = np.asarray(data.geom_xpos[l_fg, :2], dtype=np.float32)
    support = 0.5 * (right + left)
    return com - support - np.array([STAND_COM_X, 0.0], dtype=np.float32)


def balance_delta(err_b: np.ndarray, d_b: np.ndarray, squat: float = 0.0) -> np.ndarray:
    """Ankle/hip inverted-pendulum stabilizer around desired squat/stand pose."""
    err_x, err_y = float(err_b[0]), float(err_b[1])
    d_x, d_y = float(d_b[0]), float(d_b[1])
    scale = float(np.clip(1.0 - 0.5 * squat, 0.5, 1.0))
    ak_lim = float(np.clip(0.14 + 0.8 * abs(err_x), 0.14, 0.28))
    hy_lim = float(np.clip(0.20 + 1.5 * abs(err_x), 0.20, 0.50))
    sag_ak = float(np.clip(2.4 * err_x - 5.0 * d_x, -ak_lim, ak_lim)) * scale
    sag_hy = float(np.clip(4.0 * err_x - 8.0 * d_x, -hy_lim, hy_lim)) * scale
    pd_hx = float(np.clip(1.2 * err_y - 6.0 * d_y, -0.25, 0.25))
    dlt = np.zeros(N_ACT, dtype=np.float32)
    dlt[R_AP] = dlt[L_AP] = sag_ak
    dlt[R_HP] = dlt[L_HP] = sag_hy
    dlt[R_HR] = dlt[L_HR] = pd_hx
    return dlt * 0.28


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    ux, uy, uz = axis
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def extract_proprio(
    data: mujoco.MjData,
    pelvis_id: int,
    torso_id: int,
    wrist_r_id: int,
    qadr: np.ndarray,
    right_grip: float,
) -> np.ndarray:
    """28-D strictly proprioceptive robot state (zero simulator object cheats)."""
    grav, gyro = torso_imu(data, torso_id)
    q = np.asarray(data.qpos[qadr[:N_ACT]], dtype=np.float32)
    w_p = np.asarray(data.xpos[wrist_r_id], dtype=np.float64)
    w_m = np.asarray(data.xmat[wrist_r_id], dtype=np.float64).reshape(3, 3)
    hook_w = w_p + w_m @ HOOK_LOCAL_R
    pelvis_p = np.asarray(data.xpos[pelvis_id], dtype=np.float64)
    yaw = heading_z(data.qpos[3:7])
    rel_body = body_xy((hook_w[:2] - pelvis_p[:2]).astype(np.float32), yaw)
    hook_rel = np.array([rel_body[0], rel_body[1], float(hook_w[2] - pelvis_p[2])], dtype=np.float32)
    v_fwd = float(body_xy(np.asarray(data.qvel[0:2], dtype=np.float32), yaw)[0])

    vec = np.concatenate([
        grav,                          # 3
        gyro * 0.25,                   # 3
        np.array([q[WAIST_Y], q[WAIST_P]], dtype=np.float32),  # 2
        q[22:29],                      # 7 (right arm)
        q[15:22],                      # 7 (left arm)
        hook_rel,                      # 3 (right hook in body frame)
        np.array([float(right_grip), float(pelvis_p[2]), v_fwd], dtype=np.float32),  # 3
    ]).astype(np.float32)
    out = np.zeros(SLEEP_PROPRIO_DIM, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def solve_whole_body_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scratch: mujoco.MjData,
    qadr: np.ndarray,
    vadr: np.ndarray,
    wrist_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    *,
    hand: str = "right",
    squat: float = 0.0,
    q_init: np.ndarray | None = None,
    use_waist: bool = True,
) -> tuple[np.ndarray, float]:
    """6D damped-least-squares Jacobian IK over waist(yaw, pitch) + 7-DoF arm."""
    scratch.qpos[:] = data.qpos
    sq = float(np.clip(squat, 0.0, 0.80))
    q_body = ((1.0 - sq) * STAND_Q + sq * SQUAT_Q).astype(np.float32)
    if q_init is not None:
        q_body[12:15] = q_init[12:15]
        q_body[15:29] = q_init[15:29]
    for i in range(12, N_ACT):
        scratch.qpos[qadr[i]] = float(q_body[i])

    arm_idx = list(range(22, 29)) if hand != "left" else list(range(15, 22))
    act_idx = ([12, 14] + arm_idx) if use_waist else arm_idx
    hook_local = HOOK_LOCAL_R if hand != "left" else HOOK_LOCAL_L

    lo = np.array([model.jnt_range[model.actuator_trnid[i, 0], 0] for i in act_idx], dtype=np.float64)
    hi = np.array([model.jnt_range[model.actuator_trnid[i, 0], 1] for i in act_idx], dtype=np.float64)
    if use_waist:
        lo[1], hi[1] = -0.18, 0.25

    cols = np.array([vadr[i] for i in act_idx], dtype=np.int32)
    q_sub = np.array([scratch.qpos[qadr[i]] for i in act_idx], dtype=np.float64)
    weights = np.array(([0.30, 0.18] if use_waist else []) + [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2], dtype=np.float64)
    W = np.diag(weights)
    eye6 = 0.008 * np.eye(6, dtype=np.float64)

    for _ in range(45):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        w_pos = scratch.xpos[wrist_id]
        w_mat = scratch.xmat[wrist_id].reshape(3, 3)
        hook_pos = w_pos + w_mat @ hook_local
        err_p = target_pos - hook_pos
        err_r = 0.5 * (
            np.cross(w_mat[:, 0], target_mat[:, 0])
            + np.cross(w_mat[:, 1], target_mat[:, 1])
            + np.cross(w_mat[:, 2], target_mat[:, 2])
        )
        if np.linalg.norm(err_p) < 0.002 and np.linalg.norm(err_r) < 0.025:
            break
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, scratch, jacp, jacr, hook_pos, wrist_id)
        J = np.vstack([jacp[:, cols], 0.38 * jacr[:, cols]])
        err = np.concatenate([err_p, 0.38 * err_r])
        JW = J @ W
        dq = W @ JW.T @ np.linalg.solve(JW @ JW.T + eye6, err)
        q_sub = np.clip(q_sub + np.clip(dq, -0.22, 0.22), lo, hi)
        for k, idx in enumerate(act_idx):
            scratch.qpos[qadr[idx]] = float(q_sub[k])

    mujoco.mj_kinematics(model, scratch)
    hook_final = scratch.xpos[wrist_id] + scratch.xmat[wrist_id].reshape(3, 3) @ hook_local
    for k, idx in enumerate(act_idx):
        q_body[idx] = float(q_sub[k])
    return q_body, float(np.linalg.norm(target_pos - hook_final))


class System0Controller:
    """Free-standing G1 torque-PD + CoM stabilizer + RL walking gait."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, qadr: np.ndarray, vadr: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.model = model
        self.data = data
        self.qadr = qadr
        self.vadr = vadr
        self.lo = lo
        self.hi = hi
        self.pelvis_id = int(model.body("pelvis").id)
        self.torso_id = int(model.body("torso_link").id)
        self.wrist_r_id = int(model.body("right_wrist_yaw_link").id)
        self.wrist_l_id = int(model.body("left_wrist_yaw_link").id)
        self.r_fg = int(model.body("right_ankle_roll_link").id)
        self.l_fg = int(model.body("left_ankle_roll_link").id)

        # Pick box sole geom if present for accurate foot midpoint
        for side, attr in (("right", "r_fg"), ("left", "l_fg")):
            bid = int(model.body(f"{side}_ankle_roll_link").id)
            for g in range(int(model.ngeom)):
                if int(model.geom_bodyid[g]) == bid and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                    setattr(self, attr, g)
                    break

        self.walk: G1LegPolicy | None = load_g1_walk()
        self.standup: G1StandUpPolicy | None = load_g1_standup()
        self.flip: FlipPolicy | None = load_g1_flip()
        self.flipping = False
        self._flip_q = STAND_Q.copy()
        self._flip_sub = 0
        self.recovering = False
        self._rec_ticks = 0
        self._rec_stable_ticks = 0
        self._rec_blend_ticks = 0
        self.recover_cooldown = 0
        self.q_cmd = STAND_Q.copy()
        self._off_prev = np.zeros(2, dtype=np.float32)
        self._tick = 0
        self.finger_qadr = self._finger_addrs(model)[0]
        self.finger_vadr = self._finger_addrs(model)[1]
        self.finger_lo = model.actuator_ctrlrange[N_ACT:, 0].astype(np.float32)
        self.finger_hi = model.actuator_ctrlrange[N_ACT:, 1].astype(np.float32)

    @staticmethod
    def _finger_addrs(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
        qa = np.zeros(int(model.nu) - N_ACT, dtype=np.int32)
        va = np.zeros(int(model.nu) - N_ACT, dtype=np.int32)
        for i in range(N_ACT, int(model.nu)):
            jid = int(model.actuator_trnid[i, 0])
            qa[i - N_ACT] = int(model.jnt_qposadr[jid])
            va[i - N_ACT] = int(model.jnt_dofadr[jid])
        return qa, va

    def reset_pose(self, settle_ticks: int = 120) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = SPAWN_XY
        self.data.qpos[2] = SPAWN_Z
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(N_ACT):
            self.data.qpos[self.qadr[i]] = float(STAND_Q[i])
        fq = finger_pose(right_grip=0.0, tuck_left=True, tuck_right=True)
        for i, val in enumerate(fq):
            if i < len(self.finger_qadr):
                self.data.qpos[self.finger_qadr[i]] = float(val)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.q_cmd = STAND_Q.copy()
        self._off_prev[:] = 0.0
        self._tick = 0
        self.flipping = False
        self._flip_sub = 0
        self.recovering = False
        self._rec_ticks = 0
        self._rec_stable_ticks = 0
        self._rec_blend_ticks = 0
        self.recover_cooldown = 0
        if self.walk is not None:
            self.walk.reset()
        if self.standup is not None:
            self.standup.reset(self.data, self.qadr, self.vadr)
        stand = default_raw_action()
        for _ in range(settle_ticks):
            self.step_physics(stand, tuck_right=True)

    def start_standup(self) -> bool:
        """Engage the 50 Hz RL fall-recovery controller from the current fallen pose."""
        if self.standup is None:
            return False
        self.standup.reset(self.data, self.qadr, self.vadr)
        self.standup.update(self.data, self.qadr, self.vadr)
        if self.walk is not None:
            self.walk.reset()
        self.recovering = True
        self._rec_ticks = 0
        self._rec_stable_ticks = 0
        self._rec_blend_ticks = 0
        return True

    def stop_standup(self) -> None:
        self.recovering = False
        self._rec_ticks = 0
        self._rec_stable_ticks = 0
        self._rec_blend_ticks = 0
        if self.walk is not None:
            self.walk.reset()
        self.q_cmd = np.clip(np.asarray(self.data.qpos[self.qadr[:N_ACT]], dtype=np.float32), self.lo, self.hi)
        self._off_prev[:] = 0.0

    def fell(self) -> bool:
        z = float(self.data.xpos[self.pelvis_id, 2])
        tilt = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        return z < FALL_Z or tilt < TILT_LIM

    def hook_pos(self, hand: str = "right") -> np.ndarray:
        wid = self.wrist_r_id if hand != "left" else self.wrist_l_id
        hl = HOOK_LOCAL_R if hand != "left" else HOOK_LOCAL_L
        return self.data.xpos[wid] + self.data.xmat[wid].reshape(3, 3) @ hl

    def _raw_to_qdes(self, raw: np.ndarray) -> np.ndarray:
        sq = float(np.clip(raw[2], 0.0, 0.75))
        q = ((1.0 - sq) * STAND_Q + sq * SQUAT_Q).astype(np.float32)
        q[WAIST_Y] = float(raw[3])
        q[WAIST_P] = float(raw[4])
        q[WAIST_R] = 0.0
        q[22:29] = raw[5:12]
        q[15:22] = raw[12:19]
        return q

    def _step_fingers(self, right_grip: float, tuck_left: bool, tuck_right: bool) -> None:
        fq_tgt = finger_pose(
            right_grip=right_grip,
            tuck_left=tuck_left,
            tuck_right=tuck_right and right_grip < 0.05,
        )
        fq = np.asarray(self.data.qpos[self.finger_qadr], dtype=np.float32)
        fd = np.asarray(self.data.qvel[self.finger_vadr], dtype=np.float32)
        fbias = np.asarray(self.data.qfrc_bias[self.finger_vadr], dtype=np.float32)
        f_tau = FINGER_KP * (fq_tgt - fq) - FINGER_KD * fd + fbias
        self.data.ctrl[N_ACT:] = np.clip(f_tau, self.finger_lo, self.finger_hi)

    def _step_standup_tick(self) -> None:
        """Advance one 2 ms physics tick under SuperMimic RL fall-recovery control."""
        assert self.standup is not None
        blend_w = min(1.0, self._rec_blend_ticks / 350.0) if self._rec_blend_ticks > 0 else 0.0
        self.data.ctrl[:N_ACT] = self.standup.torques(self.data, self.qadr, self.vadr, KP, KD, blend_w=blend_w)
        self._step_fingers(0.0, tuck_left=True, tuck_right=True)
        mujoco.mj_step(self.model, self.data)
        self._tick += 1
        self._rec_ticks += 1

        if self._rec_blend_ticks == 0 and (self._rec_ticks % DECIMATION == 0):
            self.standup.update(self.data, self.qadr, self.vadr)

        pz = float(self.data.xpos[self.pelvis_id, 2])
        tilt = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        vel_norm = float(np.linalg.norm(self.data.qvel[:6]))
        if self._rec_blend_ticks > 0:
            self._rec_blend_ticks += 1
            if self._rec_blend_ticks > 350:
                self.q_cmd = STAND_Q.copy()
                self._off_prev[:] = 0.0
                if self.walk is not None:
                    self.walk.reset()
                self.recovering = False
                self.recover_cooldown = 500
        elif pz > 0.72 and tilt > 0.94 and vel_norm < 0.45:
            self._rec_stable_ticks += 1
            if self._rec_stable_ticks >= 250:
                self._rec_blend_ticks = 1
        else:
            self._rec_stable_ticks = 0

        if self._rec_ticks >= 4000 and self.recovering:
            self.stop_standup()
            self.recover_cooldown = 500

    def start_flip(self) -> bool:
        """Start pretrained 50 Hz ONNX backflip tracker at the current XY position."""
        if self.flip is None:
            return False
        self.recovering = False
        here = (float(self.data.qpos[0]), float(self.data.qpos[1]))
        self.flip.reset(self.model, self.data, self.qadr, self.vadr, xy=here)
        self._flip_q = self.flip.joint_pos[0].copy()
        self._flip_sub = 0
        if self.walk is not None:
            self.walk.reset()
        self.flipping = True
        return True

    def _step_flip_tick(self) -> None:
        """Advance one 2 ms physics tick under pretrained backflip policy."""
        assert self.flip is not None
        pol = self.flip
        if self._flip_sub % DECIMATION == 0 and not pol.done:
            self._flip_q = pol.act(self.model, self.data, self.qadr, self.vadr)
        self._flip_sub += 1
        self.data.ctrl[:N_ACT] = flip_torque(self.model, self.data, self._flip_q, pol, self.qadr, self.vadr)
        self._step_fingers(0.0, tuck_left=True, tuck_right=True)
        mujoco.mj_step(self.model, self.data)
        self.q_cmd = self._flip_q.copy()
        self._tick += 1
        if pol.done and (self._flip_sub % DECIMATION == 0):
            self.flipping = False
            self.q_cmd = STAND_Q.copy()
            self._off_prev[:] = 0.0
            self.recover_cooldown = 120

    def step_physics(self, raw_action: np.ndarray, *, power_on: bool = True, tuck_left: bool = True, tuck_right: bool = False) -> None:
        """Advance simulation by one 2 ms physics tick under System 0 stabilization."""
        if self.recover_cooldown > 0:
            self.recover_cooldown -= 1
        if not power_on:
            if self.recovering:
                self.recovering = False
            if self.flipping:
                self.flipping = False
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
            self._tick += 1
            return
        if self.flipping and self.flip is not None:
            self._step_flip_tick()
            return
        if self.recovering and self.standup is not None:
            self._step_standup_tick()
            return

        vx, wz = float(raw_action[0]), float(raw_action[1])
        squat = float(np.clip(raw_action[2], 0.0, 0.75))
        q_target = self._raw_to_qdes(raw_action)

        # CoM upright balance prior in body coordinates
        off = com_err_xy(self.data, self.pelvis_id, self.r_fg, self.l_fg)
        d_off = off - self._off_prev
        self._off_prev = off.copy()
        yaw = heading_z(self.data.qpos[3:7])
        bal = balance_delta(body_xy(off, yaw), body_xy(d_off, yaw), squat=squat)

        q_des = np.clip(q_target + bal, self.lo, self.hi)
        delta = np.clip(q_des - self.q_cmd, -SLEW_RAD, SLEW_RAD)
        self.q_cmd = self.q_cmd + delta

        self.data.ctrl[:N_ACT] = compute_torques(self.model, self.data, self.q_cmd, KP, KD, self.qadr, self.vadr)

        cmd_v = np.array([vx, 0.0, wz], dtype=np.float32)
        arm_q = np.asarray(self.data.qpos[self.qadr[15:29]], dtype=np.float32)
        arms_out = float(np.max(np.abs(arm_q - arm_hang_cmd()))) > 0.45 or float(np.max(np.abs(q_target[15:29] - arm_hang_cmd()))) > 0.45
        planted = float(np.max(np.abs(cmd_v))) < 0.05
        dynamic = bool(arms_out and not planted)

        if self.walk is not None:
            leg = self.walk.leg_torque(self.data, self.qadr, self.vadr, cmd_v, KP, KD, PHYSICS_DT, dynamic=dynamic)
            if leg is not None:
                self.data.ctrl[:12] = leg

        self._step_fingers(float(raw_action[19]), tuck_left=tuck_left, tuck_right=tuck_right)

        mujoco.mj_step(self.model, self.data)

        if self.walk is not None:
            if (self._tick + 1) % DECIMATION == 0 and not (self.walk.holding and planted):
                self.walk.update(self.data, self.qadr, self.vadr, cmd_v, dynamic=dynamic)
            elif self.walk.holding and planted:
                self.walk.reset()

        self._tick += 1
