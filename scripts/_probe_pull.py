"""Manual Tier A pull probe."""
from __future__ import annotations

import numpy as np
import mujoco

from agent.dex3 import FINGER_KD, FINGER_KP, finger_curl
from agent.h2 import N_ACT, STAND_Q, actuator_addrs
from agent.joint_pd import compute_torques
from agent.oven_task import OvenScene, privileged_handle
from agent.oven_teacher import _rot_axis_angle
from agent.reach import solve_arm


def main() -> None:
    for sign in (1.0, -1.0):
        s = OvenScene()
        s.reset(0, randomize=False)
        s.model.jnt_stiffness[s._door_jid] = 1.0
        s.model.dof_damping[s._door_dof] = 1.0
        s.model.dof_frictionloss[s._door_dof] = 0.5
        p = privileged_handle(s.model, s.data)
        handle = p["handle_pos"]
        s._place_robot(handle[0] - 0.28, handle[1], 0.0)
        s._settle(0.4)
        fq = finger_curl("right")
        hinge = p["hinge_pos"]
        axis = p["hinge_axis"]
        radius = handle - hinge
        qa, va = actuator_addrs(s.model)
        fqadr, fvadr = qa[N_ACT:], va[N_ACT:]
        mx = 0.0
        for ang in np.linspace(0, 1.2, 50):
            R = _rot_axis_angle(axis, float(sign * ang))
            tgt = hinge + R @ radius
            # nudge target slightly toward robot so palm presses into handle
            tgt = tgt + np.array([-0.02, 0.0, 0.0])
            goal = {
                "hand": "right",
                "u": 0.5,
                "v": 0.5,
                "point": tgt,
                "body": int(s.model.body("stove_main_group_door").id),
            }
            sol, gap = solve_arm(s.model, s.data, goal, s.qadr, s.vadr)
            qdes = sol.astype(np.float32)
            qdes[:15] = STAND_Q[:15]
            for _ in range(40):
                tau = compute_torques(s.model, s.data, qdes, s.kp, s.kd, s.qadr, s.vadr)
                s.data.ctrl[:N_ACT] = tau
                q = np.asarray(s.data.qpos[fqadr], np.float32)
                qd = np.asarray(s.data.qvel[fvadr], np.float32)
                s.data.ctrl[N_ACT:] = np.clip(
                    FINGER_KP * (fq - q) - FINGER_KD * qd,
                    s.model.actuator_ctrlrange[N_ACT:, 0],
                    s.model.actuator_ctrlrange[N_ACT:, 1],
                )
                # Privileged assist: door torque when palm is near handle.
                hand = np.asarray(s.data.xpos[s.model.body("right_wrist_yaw_link").id])
                hp = privileged_handle(s.model, s.data)
                if float(np.linalg.norm(hand - hp["handle_pos"])) < 0.12:
                    s.data.qfrc_applied[s._door_dof] = float(sign) * 25.0
                else:
                    s.data.qfrc_applied[s._door_dof] = 0.0
                mujoco.mj_step(s.model, s.data)
            door = privileged_handle(s.model, s.data)["door_angle"]
            mx = max(mx, door)
        print(f"sign={sign:+.0f} max_door={mx:.3f}")


if __name__ == "__main__":
    main()
