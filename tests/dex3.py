"""Dex3-1 is a colliding hand after the 29 body actuators. The training body stays bare."""

import mujoco
import numpy as np

from agent.dex3 import finger_target
from agent.h2 import MODEL_XML, N_ACT, SPAWN_Z, STAND_Q, TRAIN_XML


def _addrs(model):
    qadr = []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        qadr.append(int(model.jnt_qposadr[jid]))
    return np.asarray(qadr)


def _tip_gap(model, data):
    thumb = data.xpos[int(model.body("right_hand_thumb_2_link").id)]
    index = data.xpos[int(model.body("right_hand_index_1_link").id)]
    return float(np.linalg.norm(thumb - index))


def test_training_body_has_no_fingers():
    model = mujoco.MjModel.from_xml_path(str(TRAIN_XML))
    assert int(model.nu) == N_ACT


def test_app_hand_collides_and_curls():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    assert int(model.nu) == N_ACT + 14
    assert model.actuator(0).name == "left_hip_pitch"
    assert model.actuator(N_ACT - 1).name == "right_wrist_yaw"
    assert model.actuator(N_ACT).name == "left_hand_thumb_0"
    palm = model.geom("right_palm_col")
    assert int(model.geom_contype[palm.id]) != 0
    assert int(model.geom_contype[model.geom("right_palm").id]) == 0

    data.qpos[2] = SPAWN_Z
    data.qpos[3] = 1.0
    data.qpos[_addrs(model)[:N_ACT]] = STAND_Q
    mujoco.mj_forward(model, data)
    open_gap = _tip_gap(model, data)
    goal = {"hand": "right", "point": np.zeros(3)}
    data.qpos[_addrs(model)[N_ACT:]] = finger_target(goal)[0:]
    # finger qpos follows actuator order, which is the same 14.
    mujoco.mj_forward(model, data)
    shut_gap = _tip_gap(model, data)
    assert shut_gap < open_gap - 0.02, (open_gap, shut_gap)

    toaster = int(model.body("toaster_main_group_main").id)
    palm_body = int(model.geom_bodyid[palm.id])
    delta = np.asarray(data.xpos[toaster]) - np.asarray(data.geom_xpos[palm.id])
    data.qpos[0:3] = data.qpos[0:3] + delta
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    hit = False
    for i in range(data.ncon):
        bodies = {
            int(model.geom_bodyid[data.contact[i].geom1]),
            int(model.geom_bodyid[data.contact[i].geom2]),
        }
        if palm_body in bodies and toaster in bodies:
            hit = True
    assert hit, "palm collision geom does not meet the toaster"


if __name__ == "__main__":
    test_training_body_has_no_fingers()
    test_app_hand_collides_and_curls()
    print("ok")
