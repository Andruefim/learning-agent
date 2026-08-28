"""Unitree H2 body: joint map, stand poses, torque PD gains, MJCF helpers.

Official MJCF is models/unitree_h2/{h2_mujoco.xml,scene.xml,meshes/} — do not edit.
App loads scene_app.xml. Motors are torque, not position. qpos tree order ≠ actuator
order (head sits before the arms); always index hinges through qadr/vadr.
Leg+foot yaw is hip_yaw ±2.827 rad. There is no ankle yaw; ankles are roll+pitch.
"""

from __future__ import annotations

import numpy as np
import mujoco

from agent.config import PARAM_KEYS, ROOT
from agent.joint_pd import kp_kd_vectors

MODEL_XML = ROOT / "models" / "unitree_h2" / "scene_app.xml"
TRAIN_XML = ROOT / "models" / "unitree_h2" / "scene_train.xml"
N_ACT = 31
QPOS_FREE = 7
# L2 command: vx, vy, wz, h_m, 14 arm targets. Not 31 joint angles.
ACTION_DIM = 18
TRIAL_FEAT = N_ACT + ACTION_DIM + len(PARAM_KEYS)
# Actuator order (see h2_mujoco.xml <actuator>).
L_HP, L_HR, L_HYA, L_KN, L_AR, L_AP = 0, 1, 2, 3, 4, 5
R_HP, R_HR, R_HYA, R_KN, R_AR, R_AP = 6, 7, 8, 9, 10, 11
WAIST_Y, WAIST_R, WAIST_P = 12, 13, 14
L_SP, L_SR, L_SY, L_EL, L_WR, L_WP, L_WY = 15, 16, 17, 18, 19, 20, 21
R_SP, R_SR, R_SY, R_EL, R_WR, R_WP, R_WY = 22, 23, 24, 25, 26, 27, 28
HEAD_P, HEAD_Y = 29, 30
# Tracker aliases: pitch/roll/yaw of the hip, ankle pitch/roll, shoulder pitch/roll.
R_HY, L_HY = R_HP, L_HP
R_HX, L_HX = R_HR, L_HR
R_HZ, L_HZ = R_HYA, L_HYA
R_AK, L_AK = R_AP, L_AP
R_AZ, L_AZ = R_AR, L_AR
R_SH, L_SH = R_SP, L_SP
R_SX, L_SX = R_SR, L_SR
HIP_YAW_LIM = 1.0
SPAWN_Z = 1.02
ARM_RAISE = -1.55
# CAD q=0 on the arms is not a hang: the forearm body sits along +X, so zeros
# reach ~90° forward and shove CoM past the toes. Hang = positive shoulder
# pitch (arms back) + bent elbow so the hands sit beside the hips.
ARM_HANG_SP = 0.30
ARM_HANG_EL = 1.40
ARM_HANG_SR = 0.18
# Upright stand: CoM sits nearly over the foot boxes.
STAND_COM_X = 0.018


def arm_hang_cmd() -> np.ndarray:
    """14-D arm command (actuator order L then R). Stand / idle default."""
    q = np.zeros(14, dtype=np.float32)
    q[0] = ARM_HANG_SP
    q[1] = ARM_HANG_SR
    q[3] = ARM_HANG_EL
    q[7] = ARM_HANG_SP
    q[8] = -ARM_HANG_SR
    q[10] = ARM_HANG_EL
    return q


def _h2_leg_pose(*, hip: float, knee: float, ankle: float) -> np.ndarray:
    q = np.zeros(N_ACT, dtype=np.float32)
    q[L_HP] = q[R_HP] = hip
    q[L_KN] = q[R_KN] = knee
    q[L_AP] = q[R_AP] = ankle
    return q


def _apply_arm_hang(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q[15:29] = arm_hang_cmd()
    return q


STAND_Q = _apply_arm_hang(_h2_leg_pose(hip=-0.20, knee=0.40, ankle=-0.25))
STAND_Q[WAIST_P] = 0.05
# Slight hip abduction so the support polygon is not a knife-edge.
STAND_Q[L_HR] = 0.12
STAND_Q[R_HR] = -0.12
SQUAT_Q = _apply_arm_hang(_h2_leg_pose(hip=-0.55, knee=1.00, ankle=-0.40))
SQUAT_Q[WAIST_P] = 0.12
# Software Joint-PD gains (Nm/rad, Nm·s/rad). Motors stay torque in XML.
KP, KD = kp_kd_vectors()


def actuator_addrs(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos/qvel indices per actuator. Tree order ≠ motor order."""
    qadr = np.zeros(int(model.nu), dtype=np.int32)
    vadr = np.zeros(int(model.nu), dtype=np.int32)
    for i in range(int(model.nu)):
        jid = int(model.actuator_trnid[i, 0])
        qadr[i] = int(model.jnt_qposadr[jid])
        vadr[i] = int(model.jnt_dofadr[jid])
    return qadr, vadr


def joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(int(model.nu), dtype=np.float32)
    hi = np.zeros(int(model.nu), dtype=np.float32)
    for i in range(int(model.nu)):
        jid = int(model.actuator_trnid[i, 0])
        lo[i] = float(model.jnt_range[jid, 0])
        hi[i] = float(model.jnt_range[jid, 1])
    return lo, hi


def colliding_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    out: list[int] = []
    for g in range(int(model.ngeom)):
        if int(model.geom_bodyid[g]) != int(body_id):
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        out.append(int(g))
    return out


def disable_foot_spheres(model: mujoco.MjModel, body_ids: tuple[int, ...]) -> int:
    """Official H2 puts contact spheres below the sole box; they roll. Stand on the boxes."""
    n = 0
    want = {int(b) for b in body_ids}
    for g in range(int(model.ngeom)):
        if int(model.geom_bodyid[g]) not in want:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
            continue
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0
        n += 1
    return n


def disable_mesh_contacts(model: mujoco.MjModel) -> int:
    """MJX-JAX does not like mesh contacts; keep capsules/boxes for training."""
    n = 0
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    for g in range(int(model.ngeom)):
        if int(model.geom_type[g]) != mesh:
            continue
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0
        n += 1
    return n


def mjx_rocm_options(model: mujoco.MjModel) -> None:
    """Avoid hipSOLVER FFI. WSL overlays host HSA, so jax_rocm7_plugin's
    `rocm_plugin_extension.so` fails (`hsa_amd_vmem_export_fabric_handle`) and
    `cho_factor` dies with hipsolver_potrf_ffi. Dense MJX `factor_m` uses that
    path; sparse LDL + Euler + CG stay in pure XLA.
    """
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE


def cylinders_to_capsules(model: mujoco.MjModel) -> int:
    """Official H2 shins are cylinders; MJX-JAX has no cylinder-box contacts."""
    n = 0
    cyl = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    cap = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    for g in range(int(model.ngeom)):
        if int(model.geom_type[g]) != cyl:
            continue
        model.geom_type[g] = cap
        n += 1
    return n


def box_geom(model: mujoco.MjModel, geom_ids: list[int], fallback: int) -> int:
    for g in geom_ids:
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            return int(g)
    return int(fallback)
