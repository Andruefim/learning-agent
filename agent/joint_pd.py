"""Software Joint-PD on top of H2 torque motors.

`data.ctrl` is Newton-metres. Never write joint angles into ctrl.
qpos tree order ≠ actuator order (head before arms); always use qadr/vadr.
"""

from __future__ import annotations

import numpy as np

# Nm/rad and Nm·s/rad. Proven H2 stance PD (same law as before; ctrl is torque).
DEFAULT_KP = {
    "hip_pitch": 500.0,
    "hip_roll": 500.0,
    "hip_yaw": 400.0,
    "knee": 500.0,
    "ankle_pitch": 400.0,
    "ankle_roll": 80.0,
    "waist": 400.0,
    "shoulder": 80.0,
    "elbow": 80.0,
    "wrist": 10.0,
    "head": 10.0,
}
DEFAULT_KD = {
    "hip_pitch": 12.0,
    "hip_roll": 12.0,
    "hip_yaw": 8.0,
    "knee": 12.0,
    "ankle_pitch": 5.0,
    "ankle_roll": 3.0,
    "waist": 10.0,
    "shoulder": 4.0,
    "elbow": 4.0,
    "wrist": 0.5,
    "head": 0.5,
}


def kp_kd_vectors() -> tuple[np.ndarray, np.ndarray]:
    kp_d, kd_d = DEFAULT_KP, DEFAULT_KD
    hip = [
        kp_d["hip_pitch"],
        kp_d["hip_roll"],
        kp_d["hip_yaw"],
        kp_d["knee"],
        kp_d["ankle_roll"],
        kp_d["ankle_pitch"],
    ]
    hip_d = [
        kd_d["hip_pitch"],
        kd_d["hip_roll"],
        kd_d["hip_yaw"],
        kd_d["knee"],
        kd_d["ankle_roll"],
        kd_d["ankle_pitch"],
    ]
    arm = [
        kp_d["shoulder"],
        kp_d["shoulder"],
        kp_d["shoulder"],
        kp_d["elbow"],
        kp_d["wrist"],
        kp_d["wrist"],
        kp_d["wrist"],
    ]
    arm_d = [
        kd_d["shoulder"],
        kd_d["shoulder"],
        kd_d["shoulder"],
        kd_d["elbow"],
        kd_d["wrist"],
        kd_d["wrist"],
        kd_d["wrist"],
    ]
    kp = np.array(
        hip + hip + [kp_d["waist"]] * 3 + arm + arm + [kp_d["head"]] * 2,
        dtype=np.float32,
    )
    kd = np.array(
        hip_d + hip_d + [kd_d["waist"]] * 3 + arm_d + arm_d + [kd_d["head"]] * 2,
        dtype=np.float32,
    )
    return kp, kd


def compute_torques(model, data, q_target: np.ndarray, kp_vec: np.ndarray, kd_vec: np.ndarray, qadr: np.ndarray, vadr: np.ndarray) -> np.ndarray:
    """tau = Kp (q_target − q) − Kd qvel, then clip to factory actuator limits.

    Gravity/Coriolis (qfrc_bias) is added so the PD tracks pose instead of
    fighting mg. That is still a torque command, not a position servo.
    """
    q = np.asarray(data.qpos[qadr], dtype=np.float32)
    qd = np.asarray(data.qvel[vadr], dtype=np.float32)
    tgt = np.asarray(q_target, dtype=np.float32)
    tau = kp_vec * (tgt - q) - kd_vec * qd
    tau = tau + np.asarray(data.qfrc_bias[vadr], dtype=np.float32)
    lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32)
    hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32)
    return np.clip(tau, lo, hi)
