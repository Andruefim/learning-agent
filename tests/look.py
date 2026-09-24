"""A head-image pixel is a world point. Object names are not."""

import mujoco
import numpy as np

from agent.h2 import MODEL_XML, SPAWN_Z, STAND_Q
from agent.reach import (
    LOOK_H,
    LOOK_W,
    TOUCH_M,
    body_can_move,
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
from agent.reach import _arm_penetration


def _stand():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = SPAWN_Z
    data.qpos[3] = 1.0
    for i in range(len(STAND_Q)):
        jid = int(model.actuator_trnid[i, 0])
        data.qpos[int(model.jnt_qposadr[jid])] = STAND_Q[i]
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
    point, body = hit
    assert body == int(model.body("toaster_main_group_main").id)
    assert float(np.linalg.norm(point - center)) < 0.25, np.linalg.norm(point - center)


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


def test_the_arm_does_not_enter_the_counter():
    model, data = _stand()
    qadr, vadr = _addrs(model)
    toaster = int(model.body("toaster_main_group_main").id)
    target = np.asarray(data.xpos[toaster], dtype=np.float64).copy()
    spawn = np.asarray(data.qpos[:2], dtype=np.float64).copy()
    direction = target[:2] - spawn
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

    def place(distance: float) -> None:
        data.qpos[0] = float(target[0] - direction[0] * distance)
        data.qpos[1] = float(target[1] - direction[1] * distance)
        yaw = float(np.arctan2(direction[1], direction[0]))
        data.qpos[3] = float(np.cos(yaw / 2.0))
        data.qpos[6] = float(np.sin(yaw / 2.0))
        mujoco.mj_forward(model, data)

    def solve(distance: float):
        place(distance)
        goal = {"hand": "both", "u": 0.5, "v": 0.5, "point": target, "body": toaster}
        sol, gap = solve_arm(model, data, goal, qadr, vadr)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.qpos[qadr[15:29]] = sol[15:29]
        mujoco.mj_forward(model, scratch)
        pen = _arm_penetration(model, scratch, ("left", "right"), toaster)
        return goal, gap, pen

    goal, _gap, pen = solve(0.42)
    assert pen < 1e-3, pen
    near = goal.get("shift")
    if isinstance(near, np.ndarray):
        assert float(np.dot(np.asarray(near)[:2], direction)) <= 0.0, near

    goal, _gap, pen = solve(1.2)
    assert pen < 1e-3, pen
    far = goal.get("shift")
    assert isinstance(far, np.ndarray), far
    assert float(np.dot(np.asarray(far)[:2], direction)) > 0.0, far
    here = np.asarray(data.qpos[:2], dtype=np.float64)
    left = float(np.linalg.norm(target[:2] - (here + np.asarray(far)[:2])))
    assert left > TOUCH_M, left


def test_a_free_toaster_stays_on_the_counter():
    model, data = _stand()
    body = int(model.body("toaster_main_group_main").id)
    assert body_can_move(model, body)
    assert not body_can_move(model, int(model.body("stack_4_main_group_2_main").id))
    z0 = float(data.xpos[body][2])
    hold = data.qpos[:7].copy()
    for _ in range(500):
        data.qpos[:7] = hold
        data.qvel[:6] = 0.0
        mujoco.mj_step(model, data)
    z1 = float(data.xpos[body][2])
    assert abs(z1 - z0) < 0.05, (z0, z1)


if __name__ == "__main__":
    test_name_is_not_a_point()
    test_pixel_hits_the_body_in_frame()
    test_far_point_is_outside_the_arm()
    test_near_point_is_inside_the_arm()
    test_velocity_faces_the_point()
    test_a_stopped_pelvis_ends_the_walk()
    test_sleep_wants_a_closer_palm()
    test_the_arm_does_not_enter_the_counter()
    test_a_free_toaster_stays_on_the_counter()
    print("ok")
