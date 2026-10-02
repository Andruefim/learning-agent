"""A head-image pixel is a world point. Object names are not."""

import mujoco
import numpy as np

from agent.h2 import MODEL_XML, SPAWN_Z, STAND_Q
from agent.plan import Plan
from agent.planner import Level1Planner
from agent.reach import (
    LOOK_H,
    LOOK_W,
    TOUCH_M,
    body_can_move,
    camera_basis,
    cast_grasp_uv,
    goal_distance,
    merge_grasp_point,
    plan_wants_grasp_point,
    ray_direction,
    ray_hit,
    reach_goal_from_step,
    solve_arm,
    strip_planner_pixels,
    touch_lesson,
    velocity_toward,
    walk_is_blocked,
)
from agent.reach import _arm_penetration


def _stand():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    from agent.h2 import SPAWN_XY

    data.qpos[0:3] = (float(SPAWN_XY[0]), float(SPAWN_XY[1]), SPAWN_Z)
    data.qpos[3] = 1.0
    for i in range(len(STAND_Q)):
        jid = int(model.actuator_trnid[i, 0])
        data.qpos[int(model.jnt_qposadr[jid])] = STAND_Q[i]
    mujoco.mj_forward(model, data)
    return model, data


def test_main_plan_strips_pixels():
    raw = (
        '{"instruction":"поднять тостер","queue":['
        '{"hand_goal":{"hand":"both","u":0.5,"v":0.5},"vx":0,"hold_s":8}],"done":false}'
    )
    plan = Level1Planner()._parse(raw, "поднять тостер")
    assert plan is not None
    g = reach_goal_from_step(plan.queue[0])
    assert g.get("unresolved") is True and "u" not in g


def test_merge_and_cast_grasp():
    model, data = _stand()
    pelvis = int(model.body("pelvis").id)
    motion = Plan(
        instruction="поднять тостер",
        skill="hold",
        params={},
        done=False,
        queue=[{"hand_goal": {"hand": "both"}, "vx": 0, "hold_s": 8}],
    )
    assert plan_wants_grasp_point(motion)
    body = int(model.body("toaster_main_group_main").id)
    center = np.asarray(data.xpos[body], dtype=np.float64)
    origin, rot, fovy = camera_basis(model, data)
    delta = center - origin
    x, y, z = float(np.dot(delta, rot[:, 0])), float(np.dot(delta, rot[:, 1])), float(np.dot(delta, rot[:, 2]))
    aspect = LOOK_W / LOOK_H
    fov_y = float(np.deg2rad(fovy))
    fov_x = 2.0 * float(np.arctan(np.tan(fov_y / 2.0) * aspect))
    fwd = -z
    u = 0.5 + (x / fwd) / (2.0 * np.tan(fov_x / 2.0))
    v = 0.5 - (y / fwd) / (2.0 * np.tan(fov_y / 2.0))
    point_plan = Plan(
        instruction="поднять тостер",
        skill="hold",
        params={"hand_goal": {"hand": "both", "u": u, "v": v}},
        done=False,
        queue=[{"hand_goal": {"hand": "both", "u": u, "v": v}, "vx": 0, "hold_s": 1}],
    )
    assert merge_grasp_point(motion, point_plan)
    ok, hit = cast_grasp_uv(model, data, pelvis, u, v)
    assert ok and "toast" in model.body(int(hit)).name.lower()
    center_ok, _ = cast_grasp_uv(model, data, pelvis, 0.5, 0.5)
    assert center_ok
    assert not cast_grasp_uv(model, data, pelvis, 0.02, 0.5)[0]


def test_look_errors_sign():
    """Waist sense: desired = current - err centers the head camera."""
    model, data = _stand()
    body = int(model.body("toaster_main_group_main").id)
    point = np.asarray(data.xpos[body], dtype=np.float64)
    from agent.reach import look_errors

    err0 = look_errors(model, data, point)
    assert err0 is not None
    yaw0, pitch0 = err0
    # Toaster is right of spawn center → positive yaw error.
    assert yaw0 > 0.05, yaw0
    yp = int(model.jnt_qposadr[model.joint("waist_yaw_joint").id])
    data.qpos[yp] = float(data.qpos[yp]) - yaw0
    mujoco.mj_forward(model, data)
    err1 = look_errors(model, data, point)
    assert err1 is not None
    assert abs(err1[0]) < abs(yaw0)


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


def test_spawn_center_ray_hits_kitchen_not_wall():
    """Upright+tilted head at SPAWN: image center is kitchen, not the far wall."""
    model, data = _stand()
    origin, rot, fovy = camera_basis(model, data)
    aspect = LOOK_W / LOOK_H
    direction = ray_direction(rot, fovy, aspect, 0.5, 0.5)
    hit = ray_hit(model, data, origin, direction, int(model.body("pelvis").id))
    assert hit is not None
    name = model.body(int(hit[1])).name.lower()
    assert "wall" not in name, name


def test_toaster_pixel_in_frame_hits_toaster():
    """Toaster is off-center in the head image; the correct UV must hit it."""
    model, data = _stand()
    body = int(model.body("toaster_main_group_main").id)
    center = np.asarray(data.xpos[body], dtype=np.float64)
    origin, rot, fovy = camera_basis(model, data)
    delta = center - origin
    x = float(np.dot(delta, rot[:, 0]))
    y = float(np.dot(delta, rot[:, 1]))
    z = float(np.dot(delta, rot[:, 2]))
    assert z < 0.0
    aspect = LOOK_W / LOOK_H
    fov_y = float(np.deg2rad(fovy))
    fov_x = 2.0 * float(np.arctan(np.tan(fov_y / 2.0) * aspect))
    fwd = -z
    u = 0.5 + (x / fwd) / (2.0 * np.tan(fov_x / 2.0))
    v = 0.5 - (y / fwd) / (2.0 * np.tan(fov_y / 2.0))
    assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    # Not forced to image center — toaster sits to the right of stove.
    assert abs(u - 0.5) > 0.05 or abs(v - 0.5) > 0.02, (u, v)
    hit = ray_hit(model, data, origin, ray_direction(rot, fovy, aspect, u, v), int(model.body("pelvis").id))
    assert hit is not None
    assert "toast" in model.body(int(hit[1])).name.lower()


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
