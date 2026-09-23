"""A head-image pixel is a world point. Object names are not."""

import mujoco
import numpy as np

from agent.h2 import MODEL_XML, SPAWN_Z, STAND_Q
from agent.reach import (
    LOOK_H,
    LOOK_W,
    TOUCH_M,
    camera_basis,
    goal_distance,
    ray_direction,
    ray_hit,
    reach_goal_from_step,
    solve_arm,
    touch_lesson,
    velocity_toward,
    walk_is_blocked,
)


def _stand():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = SPAWN_Z
    data.qpos[3] = 1.0
    data.qpos[7 : 7 + len(STAND_Q)] = STAND_Q
    mujoco.mj_forward(model, data)
    return model, data


def test_name_is_not_a_point():
    named = reach_goal_from_step({"hand_goal": {"hand": "both", "target": "toaster"}})
    assert named.get("unresolved") is True and "u" not in named
    goal = reach_goal_from_step({"hand_goal": {"hand": "right", "u": 0.2, "v": 0.3}})
    assert goal == {"hand": "right", "u": 0.2, "v": 0.3}
    assert reach_goal_from_step({"hand_goal": {"hand": "left", "target": "head"}})["target"] == "head"


def test_pixel_hits_the_body_in_frame():
    model, data = _stand()
    body = int(model.body("toaster_main_group_main").id)
    center = np.asarray(data.xpos[body], dtype=np.float64)
    origin, rot, fovy = camera_basis(model, data)
    delta = center - origin
    x = float(np.dot(delta, rot[:, 0]))
    y = float(np.dot(delta, rot[:, 1]))
    z = float(np.dot(delta, rot[:, 2]))
    assert z < 0.0, "toaster should be in front of the head camera"
    aspect = LOOK_W / LOOK_H
    fov_y = float(np.deg2rad(fovy))
    fov_x = 2.0 * float(np.arctan(np.tan(fov_y / 2.0) * aspect))
    fwd = -z
    u = 0.5 + (x / fwd) / (2.0 * np.tan(fov_x / 2.0))
    v = 0.5 - (y / fwd) / (2.0 * np.tan(fov_y / 2.0))
    assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    direction = ray_direction(rot, fovy, aspect, u, v)
    hit = ray_hit(model, data, origin, direction, int(model.body("pelvis").id))
    assert hit is not None
    assert float(np.linalg.norm(hit - center)) < 0.25, np.linalg.norm(hit - center)


def test_velocity_faces_the_point():
    vx, wz = velocity_toward(0.0, np.zeros(3), np.array([1.2, -0.4, 1.0]))
    assert vx > 0.2, vx
    assert wz < 0.0, wz


def _addrs(model):
    qadr, vadr = [], []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qadr), np.asarray(vadr)


def test_far_point_is_outside_the_arm():
    model, data = _stand()
    qadr, vadr = _addrs(model)
    goal = {"hand": "right", "u": 0.5, "v": 0.5, "point": np.array([1.1, -0.6, 1.0])}
    before = goal_distance(model, data, goal)
    _sol, gap = solve_arm(model, data, goal, qadr, vadr)
    assert before is not None and before > TOUCH_M
    assert gap > TOUCH_M, gap


def test_near_point_is_inside_the_arm():
    model, data = _stand()
    qadr, vadr = _addrs(model)
    palm_body = int(model.body("right_wrist_yaw_link").id)
    near = np.asarray(data.xpos[palm_body], dtype=np.float64) + np.array([0.05, -0.05, 0.15])
    goal = {"hand": "right", "u": 0.5, "v": 0.5, "point": near}
    before = goal_distance(model, data, goal)
    _sol, gap = solve_arm(model, data, goal, qadr, vadr)
    assert before is not None and gap < before
    assert gap <= TOUCH_M, (before, gap)


def test_a_stopped_pelvis_ends_the_walk():
    point = np.array([0.4, 0.0, 1.0])
    assert walk_is_blocked(0.0, np.zeros(3), np.zeros(2), point)
    assert not walk_is_blocked(0.0, np.zeros(3), np.array([0.3, 0.0]), point)
    assert not walk_is_blocked(0.0, np.zeros(3), np.zeros(2), np.array([0.2, 0.8, 1.0]))


def test_sleep_wants_a_closer_palm():
    assert touch_lesson(False, None, None) is None
    assert touch_lesson(True, None, None) is not None
    assert touch_lesson(True, 1.0, 0.97) is not None
    assert touch_lesson(True, 1.0, 0.7) is None


if __name__ == "__main__":
    test_name_is_not_a_point()
    test_pixel_hits_the_body_in_frame()
    test_far_point_is_outside_the_arm()
    test_near_point_is_inside_the_arm()
    test_velocity_faces_the_point()
    test_a_stopped_pelvis_ends_the_walk()
    test_sleep_wants_a_closer_palm()
    print("ok")
