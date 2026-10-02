"""Move a hand toward a body landmark. Joint targets come from the Jacobian, not a pose table."""

from __future__ import annotations

import numpy as np
import mujoco

# Palm point in the wrist-yaw frame. +X runs out through the rubber hand.
# Match right_palm_col (~0.0415). A larger offset left the collider short of the
# handle while the residual looked closed — pull then dragged empty air.
_PALM = np.array([0.04, 0.0, 0.0], dtype=np.float64)
_HAND_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
_ARM_ROOT = {"left": "left_shoulder_pitch_link", "right": "right_shoulder_pitch_link"}
_ARM0 = {"left": 15, "right": 22}
_CMD0 = {"left": 4, "right": 11}


def reach_goal_from_text(text: str) -> dict | None:
    """A hand and a place already on the body. No joint angles."""
    raw = str(text or "").lower().replace("ё", "е")
    if " на " not in f" {raw} ":
        return None
    if any(w in raw for w in ("левую", "левая", "левой", "левой", "left")):
        hand = "left"
    elif any(w in raw for w in ("правую", "правая", "правой", "right")):
        hand = "right"
    elif any(w in raw for w in ("руки", "обе руки", "both")):
        hand = "both"
    elif any(w in raw for w in ("руку", "рука", "hand")):
        hand = "left"
    else:
        return None
    if any(w in raw for w in ("голов", "head")):
        target = "head"
    else:
        return None
    return {"hand": hand, "target": target}


# Head image the planner sees. u,v are fractions of this frame, so the ray uses the same aspect.
LOOK_W, LOOK_H = 320, 240
# Rubber palm counts as touching when the solver residual is inside this.
TOUCH_M = 0.08
# A lesson is a palm that actually moved toward the point.
CLOSER_M = 0.05
# After the body has arrived, this long without contact means the grasp missed.
GRASP_S = 1.5
# A free body counts as lifted once contact has raised it by this much.
LIFT_M = 0.04
# How fast the aim climbs while the named hands stay on a free body.
LIFT_MPS = 0.02
# Contact has to last this long before a fixed body counts as held.
CONTACT_S = 0.4
# Give the next look this many misses, then stop the attempt.
GRASP_TRIES = 6
# Aim on the hit (or slightly into it). A positive standoff left the palm short of
# the handle while the residual looked closed.
STANDOFF_M = -0.01
# Stop the feet and raise once the palm residual is in this band — no body-distance
# plant heuristic (that clenched the hand while the arm stayed down).
RAISE_GAP_M = 0.28
# Forward / yaw channel limits while the point is outside the arm.
_VX_CAP = 0.4
_WZ_CAP = 0.8
# The body has arrived when it is told to walk forward and the pelvis does not.
ARRIVE_S = 0.5
_BLOCK_BEARING = 0.8
_BLOCK_VX = 0.12
_BLOCK_SPEED = 0.05


def _hand_name(raw: dict) -> str | None:
    hand = str(raw.get("hand") or raw.get("body") or "").strip().lower()
    hand = {"left_hand": "left", "right_hand": "right", "лев": "left", "прав": "right"}.get(hand, hand)
    return hand if hand in {"left", "right", "both"} else None


def _image_uv(raw: dict) -> tuple[float, float] | None:
    look = raw.get("look")
    if isinstance(look, (list, tuple)) and len(look) == 2:
        u, v = look
    else:
        u, v = raw.get("u"), raw.get("v")
    try:
        u_f, v_f = float(u), float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(u_f) or not np.isfinite(v_f):
        return None
    return float(np.clip(u_f, 0.0, 1.0)), float(np.clip(v_f, 0.0, 1.0))


def reach_goal_from_step(step: dict) -> dict | None:
    raw = step.get("hand_goal") if isinstance(step, dict) else None
    if not isinstance(raw, dict):
        nested = step.get("params") if isinstance(step, dict) else None
        raw = nested.get("hand_goal") if isinstance(nested, dict) else None
    if not isinstance(raw, dict):
        return None
    hand = _hand_name(raw)
    if hand is None:
        return None
    target = str(raw.get("target") or "").strip().lower()
    if target == "head":
        return {"hand": hand, "target": "head"}
    uv = _image_uv(raw)
    if uv is None:
        return {"hand": hand, "unresolved": True}
    return {"hand": hand, "u": uv[0], "v": uv[1]}


def attach_reach(plan, text: str) -> None:
    """If the sentence names a hand and a place, keep that goal on the frame the model returned."""
    goal = None
    for step in plan.queue or []:
        goal = reach_goal_from_step(step)
        if goal:
            break
    if goal is None:
        goal = reach_goal_from_text(text) or reach_goal_from_text(getattr(plan, "instruction", ""))
    if not goal:
        return
    for step in plan.queue or []:
        if str(step.get("hands") or "").lower() == "down":
            continue
        step["hand_goal"] = dict(goal)
        break
    else:
        if plan.queue:
            plan.queue[0]["hand_goal"] = dict(goal)
    plan.params = dict(plan.params or {})
    plan.params["hand_goal"] = dict(goal)


def strip_planner_pixels(plan) -> None:
    """Main L1 plan must not carry u,v — pointing is a separate vision pass."""
    for step in plan.queue or []:
        raw = step.get("hand_goal")
        if not isinstance(raw, dict):
            continue
        if str(raw.get("target") or "").strip().lower() == "head":
            continue
        raw.pop("u", None)
        raw.pop("v", None)
        raw.pop("look", None)
    params = plan.params if isinstance(plan.params, dict) else {}
    raw = params.get("hand_goal")
    if isinstance(raw, dict) and str(raw.get("target") or "").strip().lower() != "head":
        raw.pop("u", None)
        raw.pop("v", None)
        raw.pop("look", None)


def plan_wants_grasp_point(plan) -> bool:
    """Touch/grasp a visible object — needs plan_point(), not coordinates in the main plan."""
    for step in plan.queue or []:
        g = reach_goal_from_step(step)
        if g is None:
            continue
        if g.get("target") == "head":
            continue
        return True
    params = plan.params if isinstance(plan.params, dict) else {}
    raw = params.get("hand_goal")
    if isinstance(raw, dict) and _hand_name(raw):
        if str(raw.get("target") or "").strip().lower() == "head":
            return False
        return True
    return False


def merge_grasp_point(plan, point_plan) -> bool:
    """Inject hand,u,v from the vision-only pointing plan."""
    merged: dict | None = None
    for step in point_plan.queue or []:
        raw = step.get("hand_goal")
        if isinstance(raw, dict) and _image_uv(raw):
            merged = {"hand": _hand_name(raw) or "right", "u": float(raw["u"]), "v": float(raw["v"])}
            break
    if merged is None:
        params = point_plan.params if isinstance(point_plan.params, dict) else {}
        raw = params.get("hand_goal")
        if isinstance(raw, dict):
            uv = _image_uv(raw)
            if uv:
                merged = {"hand": _hand_name(raw) or "right", "u": uv[0], "v": uv[1]}
    if merged is None:
        return False
    for step in plan.queue or []:
        g = reach_goal_from_step(step)
        if g is not None and g.get("target") != "head":
            step["hand_goal"] = dict(merged)
    plan.params = dict(plan.params or {})
    plan.params["hand_goal"] = dict(merged)
    attach_reach(plan, plan.instruction)
    return True


def cast_grasp_uv(model, data, pelvis_id: int, u: float, v: float) -> tuple[bool, int | None]:
    """True when the head-camera ray at (u,v) hits a solid body, not wall/floor/miss."""
    origin, rot, fovy = camera_basis(model, data)
    direction = ray_direction(rot, fovy, LOOK_W / LOOK_H, float(u), float(v))
    hit = ray_hit(model, data, origin, direction, int(pelvis_id))
    if hit is None:
        return False, None
    _point, body = hit
    root = int(assembly_root(model, int(body)))
    name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or "?").lower()
    if "wall" in name or "floor" in name or name in {"world", "ground"}:
        return False, root
    u_f, v_f = float(u), float(v)
    if u_f < 0.12 or u_f > 0.88 or v_f < 0.12 or v_f > 0.88:
        return False, root
    return True, root


def touch_lesson(aimed: bool, d0: float | None, best: float | None) -> str | None:
    """Sleep keeps a reach only when the palm got closer. Other commands are unchanged."""
    if not aimed:
        return None
    if d0 is None or best is None:
        return "Сон пропущен: луч не попал в предмет."
    if float(d0) - float(best) < CLOSER_M:
        return "Сон пропущен: рука не приблизилась."
    return None


def camera_basis(model, data):
    cam = int(model.camera("head").id)
    origin = np.asarray(data.cam_xpos[cam], dtype=np.float64)
    rot = np.asarray(data.cam_xmat[cam], dtype=np.float64).reshape(3, 3)
    fovy = float(model.cam_fovy[cam])
    return origin, rot, fovy


def look_errors(model, data, point: np.ndarray) -> tuple[float, float] | None:
    """Yaw/pitch error (rad) of the head camera relative to a world point.

    Positive yaw_err → target is to the camera's right.
    Positive pitch_err → target is above the image center.
    """
    origin, rot, _ = camera_basis(model, data)
    delta = np.asarray(point, dtype=np.float64) - origin
    right = rot[:, 0]
    up = rot[:, 1]
    forward = -rot[:, 2]
    depth = float(np.dot(delta, forward))
    if depth < 0.05:
        return None
    yaw_err = float(np.arctan2(float(np.dot(delta, right)), depth))
    pitch_err = float(np.arctan2(float(np.dot(delta, up)), depth))
    return yaw_err, pitch_err


def ray_direction(rot: np.ndarray, fovy_deg: float, aspect: float, u: float, v: float) -> np.ndarray:
    """Camera ray for image fractions. v=0 is the top of the JPEG."""
    fov_y = float(np.deg2rad(fovy_deg))
    fov_x = 2.0 * float(np.arctan(np.tan(fov_y / 2.0) * aspect))
    right = rot[:, 0]
    up = rot[:, 1]
    forward = -rot[:, 2]
    direction = (
        forward
        + right * np.tan(fov_x / 2.0) * (float(u) - 0.5) * 2.0
        + up * np.tan(fov_y / 2.0) * (0.5 - float(v)) * 2.0
    )
    return direction / max(float(np.linalg.norm(direction)), 1e-8)


def ray_hit(model, data, origin: np.ndarray, direction: np.ndarray, bodyexclude: int) -> tuple[np.ndarray, int] | None:
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(
        model,
        data,
        np.asarray(origin, dtype=np.float64),
        np.asarray(direction, dtype=np.float64),
        None,
        1,
        int(bodyexclude),
        geomid,
    )
    if dist < 0.0:
        return None
    point = np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * float(dist)
    return point, int(model.geom_bodyid[int(geomid[0])])


def assembly_root(model, body: int) -> int:
    """The jointed body this geom moves with. A window is part of its door.

    Fixed children climb to the body that owns the joint. A hit with no joint
    stays itself, so a static counter is not the whole kitchen.
    """
    body = int(body)
    origin = body
    while True:
        adr = int(model.body_jntadr[body])
        n = int(model.body_jntnum[body])
        if adr >= 0 and n > 0:
            return body
        parent = int(model.body_parentid[body])
        if parent <= 0:
            return origin
        body = parent


def palm_near_point(model, data, hand: str, point, max_m: float) -> bool:
    """True when the named palm is within max_m of the grasp point."""
    sides = ("left", "right") if hand == "both" else (str(hand),)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    for side in sides:
        if side not in _HAND_BODY:
            continue
        palm, _ = _palm(model, data, side)
        if float(np.linalg.norm(palm - point)) <= float(max_m):
            return True
    return False


def _under(model, body: int, root: int) -> bool:
    for _ in range(64):
        if int(body) == int(root):
            return True
        if int(body) <= 0:
            return False
        body = int(model.body_parentid[int(body)])
    return False


def hand_touching(model, data, hand: str, body_id: int) -> bool:
    """Geoms of one hand are in contact with the body the ray hit."""
    name = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}.get(hand)
    if name is None or int(body_id) < 0:
        return False
    root = int(model.body(name).id)
    target = int(body_id)
    for i in range(int(data.ncon)):
        b1 = int(model.geom_bodyid[int(data.contact[i].geom1)])
        b2 = int(model.geom_bodyid[int(data.contact[i].geom2)])
        if (_under(model, b1, root) and _under(model, b2, target)) or (
            _under(model, b2, root) and _under(model, b1, target)
        ):
            return True
    return False


def hands_holding(model, data, hand: str, body_id: int) -> bool:
    """Every hand named in the plan is on the body the ray hit."""
    sides = ("left", "right") if hand == "both" else (hand,)
    sides = [side for side in sides if side in ("left", "right")]
    return bool(sides) and all(hand_touching(model, data, side, int(body_id)) for side in sides)


def body_can_move(model, body_id: int) -> bool:
    """A free joint on the hit body means the hands can pick it up."""
    adr = int(model.body_jntadr[int(body_id)])
    n = int(model.body_jntnum[int(body_id)])
    if adr < 0 or n <= 0:
        return False
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    return any(int(model.jnt_type[j]) == free for j in range(adr, adr + n))


def velocity_toward(yaw: float, origin: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    """Forward speed and yaw rate of the existing walk command, toward a world point."""
    delta = np.asarray(point[:2], dtype=np.float64) - np.asarray(origin[:2], dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        return 0.0, 0.0
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    bearing = float(np.arctan2(float(np.dot(delta, left)), float(np.dot(delta, heading))))
    facing = max(float(np.cos(bearing)), 0.0)
    vx = float(np.clip(dist, 0.0, _VX_CAP) * facing)
    wz = float(np.clip(bearing, -_WZ_CAP, _WZ_CAP))
    return vx, wz


def walk_is_blocked(yaw: float, origin: np.ndarray, velocity_xy: np.ndarray, point: np.ndarray) -> bool:
    """Forward speed is commanded toward the point, and the pelvis is not moving that way."""
    delta = np.asarray(point[:2], dtype=np.float64) - np.asarray(origin[:2], dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        return False
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    bearing = float(np.arctan2(float(np.dot(delta, left)), float(np.dot(delta, heading))))
    facing = max(float(np.cos(bearing)), 0.0)
    vx = float(np.clip(dist, 0.0, _VX_CAP) * facing)
    forward = float(np.dot(np.asarray(velocity_xy[:2], dtype=np.float64), heading))
    return abs(bearing) < _BLOCK_BEARING and vx > _BLOCK_VX and forward < _BLOCK_SPEED


def _palm(model, data, side: str) -> tuple[np.ndarray, int]:
    bid = int(model.body(_HAND_BODY[side]).id)
    rot = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    point = np.asarray(data.xpos[bid], dtype=np.float64) + rot @ _PALM
    return point, bid


def _palm_axis(model, data, side: str) -> np.ndarray:
    """World direction of the fingers (+X of the wrist-yaw frame)."""
    bid = int(model.body(_HAND_BODY[side]).id)
    rot = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    return np.asarray(rot[:, 0], dtype=np.float64).copy()


def _aim_cost(model, data, goal: dict, side: str) -> float:
    """0 when the fingers lead toward the goal, 1 when they point the opposite way.

    Measure from the wrist body, not the palm tip: aiming the tip lets the wrist
    bone press the face while the fingers hang sideways. Prefer a level aim so
    the hand does not dive down onto the bar from above.
    """
    target = _target_point(model, data, goal, side)
    if target is None:
        return 0.0
    bid = int(model.body(_HAND_BODY[side]).id)
    origin = np.asarray(data.xpos[bid], dtype=np.float64)
    delta = np.asarray(target, dtype=np.float64) - origin
    # Soften the vertical so fingers stay forward, not pointed at the floor.
    delta = np.array([delta[0], delta[1], 0.25 * delta[2]], dtype=np.float64)
    n = float(np.linalg.norm(delta))
    if n < 1e-4:
        return 0.0
    direction = delta / n
    axis = _palm_axis(model, data, side)
    return float(0.5 * (1.0 - float(np.dot(axis, direction))))


def _reach_cost(model, data, goal: dict, side: str, trial: np.ndarray, q_now: np.ndarray) -> float:
    dist = _distance(model, data, goal, side)
    # Softly prefer a quiet wrist: extreme yaw looks like opening with the bone.
    wrist = float(np.linalg.norm(trial[4:7]))
    palm, bid = _palm(model, data, side)
    origin = np.asarray(data.xpos[bid], dtype=np.float64)
    target = _target_point(model, data, goal, side)
    # Fingers dangling below the wrist bone are the "opens with the wrist" look.
    drop = float(max(0.0, origin[2] - float(palm[2]) - 0.02))
    # Wrist high above the target → overhand, bone hits first.
    above = 0.0
    if target is not None:
        above = float(max(0.0, origin[2] - float(target[2]) - 0.04))
    return (
        dist
        + 0.55 * _aim_cost(model, data, goal, side)
        + 0.04 * wrist
        + 0.35 * drop
        + 0.45 * above
        + 0.02 * float(np.linalg.norm(trial - q_now))
    )


def _orient_wrist(model, scratch, goal: dict, side: str, qadr, lo: np.ndarray, hi: np.ndarray) -> None:
    """Turn only the wrist so the fingers lead toward the point. Shoulder and elbow stay put."""
    act0 = _ARM0[side]
    base = np.asarray(scratch.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
    best = base.copy()

    def _orient_score(trial: np.ndarray) -> float:
        scratch.qpos[qadr[act0 : act0 + 7]] = trial
        _sync(model, scratch)
        dist = _distance(model, scratch, goal, side)
        palm, bid = _palm(model, scratch, side)
        origin = np.asarray(scratch.xpos[bid], dtype=np.float64)
        drop = float(max(0.0, origin[2] - float(palm[2]) - 0.02))
        target = _target_point(model, scratch, goal, side)
        above = 0.0
        if target is not None:
            above = float(max(0.0, origin[2] - float(target[2]) - 0.04))
        # Aim dominates: a near palm with fingers pointing at the floor is the
        # "opens with the wrist" failure mode we just measured live (aim≈0.6).
        return (
            1.0 * _aim_cost(model, scratch, goal, side)
            + 0.12 * dist
            + 0.25 * drop
            + 0.30 * above
            + 0.02 * float(np.linalg.norm(trial[4:7] - base[4:7]))
        )

    best_cost = _orient_score(base)
    # Full joint range — the old ±1 rad cone could not flip a downward +X to forward.
    for wr in np.linspace(float(lo[4]), float(hi[4]), 7):
        for wp in np.linspace(float(lo[5]), float(hi[5]), 7):
            for wy in np.linspace(float(lo[6]), float(hi[6]), 7):
                trial = base.copy()
                trial[4], trial[5], trial[6] = float(wr), float(wp), float(wy)
                cost = _orient_score(trial)
                if cost < best_cost:
                    best_cost = cost
                    best = trial.copy()
    scratch.qpos[qadr[act0 : act0 + 7]] = best
    _sync(model, scratch)


def _head_target(model, data, side: str) -> np.ndarray:
    """A point just beside the head camera, on the side of the named hand."""
    cam = np.asarray(data.cam_xpos[int(model.camera("head").id)], dtype=np.float64)
    rot = np.asarray(data.xmat[int(model.body("torso_link").id)], dtype=np.float64).reshape(3, 3)
    side_axis = rot[:, 1] if side == "left" else -rot[:, 1]
    return cam + side_axis * 0.11 - rot[:, 2] * 0.07


def _limits(model, act0: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(7, dtype=np.float64)
    hi = np.zeros(7, dtype=np.float64)
    for k in range(7):
        jid = int(model.actuator_trnid[act0 + k, 0])
        lo[k], hi[k] = model.jnt_range[jid]
    return lo, hi


def _clip_arm(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(q, lo, hi)


def _sync(model, scratch) -> None:
    """Body positions and the head camera. Kinematics alone leaves cam_xpos stale."""
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_camlight(model, scratch)


def _target_point(model, data, goal: dict, side: str) -> np.ndarray | None:
    if "u" in goal:
        if goal.get("point") is None:
            return None
        return np.asarray(goal["point"], dtype=np.float64)
    return _head_target(model, data, side)


def goal_distance(model, data, goal: dict) -> float | None:
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    dists = []
    for side in sides:
        if side not in _HAND_BODY:
            continue
        target = _target_point(model, data, goal, side)
        if target is None:
            continue
        palm, _ = _palm(model, data, side)
        dists.append(float(np.linalg.norm(target - palm)))
    if not dists:
        return None
    return float(min(dists))


def _distance(model, scratch, goal: dict, side: str) -> float:
    point, _ = _palm(model, scratch, side)
    return float(np.linalg.norm(_target_point(model, scratch, goal, side) - point))


def _refine(model, scratch, goal: dict, side: str, qadr, vadr, lo: np.ndarray, hi: np.ndarray) -> None:
    """Local steps after a candidate is already near the point."""
    act0 = _ARM0[side]
    for _ in range(20):
        point, bid = _palm(model, scratch, side)
        target = _target_point(model, scratch, goal, side)
        err = target - point
        dist = float(np.linalg.norm(err))
        if dist < 0.04:
            break
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, scratch, jacp, jacr, point, bid)
        cols = np.asarray(vadr[act0 : act0 + 7], dtype=np.int32)
        jac = jacp[:, cols]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.08 * np.eye(3), err)
        # Wrist DOFs are for aiming the fingers. Letting the position step twist
        # them left the live hand pressing the handle with the bone.
        dq[4:7] = 0.0
        q = _clip_arm(scratch.qpos[qadr[act0 : act0 + 7]] + np.clip(dq, -0.15, 0.15), lo, hi)
        scratch.qpos[qadr[act0 : act0 + 7]] = q
        _sync(model, scratch)
    _orient_wrist(model, scratch, goal, side, qadr, lo, hi)


def _contacts(model, scratch) -> None:
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    mujoco.mj_collision(model, scratch)


def _arm_penetration(model, scratch, sides, allow_body: int | None) -> float:
    """How far these arms enter a body that is not the robot and not the one being grasped."""
    pelvis = int(model.body("pelvis").id)
    roots = [int(model.body(_ARM_ROOT[side]).id) for side in sides if side in _ARM_ROOT]
    if not roots:
        return 0.0
    pen = 0.0
    allow = None if allow_body is None or int(allow_body) < 0 else int(allow_body)
    for i in range(int(scratch.ncon)):
        contact = scratch.contact[i]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        arm1 = any(_under(model, b1, root) for root in roots)
        arm2 = any(_under(model, b2, root) for root in roots)
        if arm1 == arm2:
            continue
        other = b2 if arm1 else b1
        if _under(model, other, pelvis):
            continue
        if allow is not None and _under(model, other, allow):
            continue
        pen += max(0.0, -float(contact.dist))
    return float(pen)


def _set_arms(scratch, qadr, arms: dict) -> None:
    for side, q in arms.items():
        act0 = _ARM0[side]
        scratch.qpos[qadr[act0 : act0 + 7]] = q


def _path_penetration(model, scratch, data, qadr, goal, current: dict, target: dict, sides, allow_body) -> float:
    worst = 0.0
    for t in (0.25, 0.5, 0.75, 1.0):
        arms = {side: (1.0 - t) * current[side] + t * target[side] for side in sides}
        scratch.qpos[:] = data.qpos
        _set_arms(scratch, qadr, arms)
        _contacts(model, scratch)
        worst = max(worst, _arm_penetration(model, scratch, sides, allow_body))
        if worst > 1e-3:
            return float(worst)
    return float(worst)


def _posed(model, scratch, data, qadr, goal, arms, sides, allow_body, pelvis_xy) -> tuple[float, float]:
    scratch.qpos[:] = data.qpos
    scratch.qpos[0] = float(pelvis_xy[0])
    scratch.qpos[1] = float(pelvis_xy[1])
    _set_arms(scratch, qadr, arms)
    _contacts(model, scratch)
    pen = _arm_penetration(model, scratch, sides, allow_body)
    dists = []
    for side in sides:
        dists.append(_distance(model, scratch, goal, side))
    dist = float(max(dists)) if dists else float("inf")
    return float(pen), dist


def _arm_span(model, side: str) -> float:
    """Longest the palm can be from the shoulder: the sum of the arm's own link lengths."""
    span = float(np.linalg.norm(_PALM))
    body = int(model.body(_HAND_BODY[side]).id)
    root = int(model.body(_ARM_ROOT[side]).id)
    for _ in range(16):
        if int(body) == int(root) or int(body) <= 0:
            break
        span += float(np.linalg.norm(np.asarray(model.body_pos[int(body)], dtype=np.float64)))
        body = int(model.body_parentid[int(body)])
    return float(span)


def _raised_clear(model, scratch, data, qadr, goal, arms, sides, allow_body, pelvis_xy) -> bool:
    pen, _dist = _posed(model, scratch, data, qadr, goal, arms, sides, allow_body, pelvis_xy)
    return pen <= 1e-3


def _stance_for_raise(
    model, scratch, data, qadr, goal, current: dict, raised: dict, sides, allow_body
) -> np.ndarray | None:
    """Pelvis offset along the line to the point where raising the hand still stays clear.

    The window is the arm's own length. The place is the last step at which the wrist
    can come up to the point without entering another body.
    """
    point = goal.get("point")
    if point is None or not raised:
        return None
    here = np.asarray(data.qpos[:2], dtype=np.float64).copy()
    delta = np.asarray(point[:2], dtype=np.float64) - here
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        return None
    forward = delta / dist
    span = max(_arm_span(model, side) for side in sides)

    def clear(step: float) -> bool:
        pelvis = here + forward * float(step)
        for t in (0.5, 1.0):
            arms = {side: (1.0 - t) * current[side] + t * raised[side] for side in sides}
            if not _raised_clear(model, scratch, data, qadr, goal, arms, sides, allow_body, pelvis):
                return False
        return True

    if clear(0.0):
        lo, hi = 0.0, dist
        if not clear(dist):
            while hi - lo > TOUCH_M:
                mid = 0.5 * (lo + hi)
                if clear(mid):
                    lo = mid
                else:
                    hi = mid
        best = dist if clear(dist) else lo
    else:
        lo, hi = -span, 0.0
        if not clear(-span):
            return None
        while hi - lo > TOUCH_M:
            mid = 0.5 * (lo + hi)
            if clear(mid):
                lo = mid
            else:
                hi = mid
        best = lo
    if abs(float(best)) < TOUCH_M:
        return None
    return forward * float(best)


def velocity_shift(yaw: float, shift: np.ndarray) -> tuple[float, float]:
    """Body-frame walk that carries the pelvis along a world shift. Backward and sideways included."""
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    delta = np.asarray(shift[:2], dtype=np.float64)
    fwd = float(np.dot(delta, heading))
    lat = float(np.dot(delta, left))
    return float(np.clip(fwd * 1.5, -0.25, 0.25)), float(np.clip(lat * 1.5, -0.25, 0.25))


def solve_arm(model, data, goal: dict, qadr, vadr) -> tuple[np.ndarray, float]:
    """Grid the shoulder and elbow, then turn the wrist so the fingers face the point."""
    sol = np.asarray(data.qpos[qadr], dtype=np.float64).copy()
    scratch = mujoco.MjData(model)
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    sides = tuple(side for side in sides if side in _HAND_BODY)
    allow = goal.get("body")
    if allow is not None and int(allow) >= 0:
        allow = int(assembly_root(model, int(allow)))
        goal["body"] = allow
    current = {}
    by_roll = {}
    kinematic = {}
    for side in sides:
        act0 = _ARM0[side]
        lo, hi = _limits(model, act0)
        q_now = np.asarray(data.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
        current[side] = q_now
        best_q = q_now.copy()
        best_cost = _distance(model, data, goal, side)
        rolls = np.linspace(lo[1], hi[1], 5)
        per_roll = {}
        for pitch in np.linspace(lo[0], hi[0], 7):
            for ri, roll in enumerate(rolls):
                for elbow in np.linspace(lo[3], hi[3], 5):
                    trial = q_now.copy()
                    trial[0], trial[1], trial[3] = pitch, roll, elbow
                    scratch.qpos[:] = data.qpos
                    scratch.qpos[qadr[act0 : act0 + 7]] = trial
                    _sync(model, scratch)
                    dist = _distance(model, scratch, goal, side)
                    cost = dist + 0.03 * float(np.linalg.norm(trial - q_now))
                    if cost < best_cost:
                        best_cost = cost
                        best_q = trial.copy()
                    prev = per_roll.get(ri)
                    if prev is None or dist < prev[0]:
                        per_roll[ri] = (dist, trial.copy())
        # Wrist was frozen in the shoulder/elbow sweep. Turn it so fingers face the
        # point before we score clearance, otherwise the wrist bone wins the contact.
        oriented = {}
        for ri, picked in per_roll.items():
            scratch.qpos[:] = data.qpos
            scratch.qpos[qadr[act0 : act0 + 7]] = picked[1]
            _sync(model, scratch)
            _orient_wrist(model, scratch, goal, side, qadr, lo, hi)
            trial = np.asarray(scratch.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
            dist = _distance(model, scratch, goal, side)
            palm, bid = _palm(model, scratch, side)
            origin = np.asarray(scratch.xpos[bid], dtype=np.float64)
            drop = float(max(0.0, origin[2] - float(palm[2]) - 0.02))
            target = _target_point(model, scratch, goal, side)
            above = 0.0
            if target is not None:
                above = float(max(0.0, origin[2] - float(target[2]) - 0.04))
            score = (
                dist
                + 0.55 * _aim_cost(model, scratch, goal, side)
                + 0.04 * float(np.linalg.norm(trial[4:7]))
                + 0.35 * drop
                + 0.45 * above
            )
            oriented[ri] = (score, trial, dist)
        by_roll[side] = oriented
        scratch.qpos[:] = data.qpos
        scratch.qpos[qadr[act0 : act0 + 7]] = best_q
        _sync(model, scratch)
        _orient_wrist(model, scratch, goal, side, qadr, lo, hi)
        kinematic[side] = np.asarray(scratch.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
        sol[act0 : act0 + 7] = q_now
    if not sides:
        goal["shift"] = None
        return sol, float("inf")

    # One candidate per shoulder roll, so a swept-back shoulder is scored even if a straight arm is closer.
    candidates = []
    roll_ids = range(5)
    for ri in roll_ids:
        arms = {}
        score = 0.0
        gap = 0.0
        ok = True
        for side in sides:
            picked = by_roll[side].get(ri)
            if picked is None:
                ok = False
                break
            score = max(score, float(picked[0]))
            gap = max(gap, float(picked[2]))
            arms[side] = picked[1]
        if ok:
            candidates.append((score, gap, arms))
    clear = None
    clear_dist = float("inf")
    clear_score = float("inf")
    for score, gap, arms in candidates:
        pen = _path_penetration(model, scratch, data, qadr, goal, current, arms, sides, allow)
        if pen <= 1e-3 and score < clear_score:
            clear = {side: q.copy() for side, q in arms.items()}
            clear_dist = gap
            clear_score = score
    kin_arms = {side: kinematic[side] for side in sides}
    kin_pen, kin_dist = _posed(model, scratch, data, qadr, goal, kin_arms, sides, allow, data.qpos[:2])
    chosen = {side: q.copy() for side, q in clear.items()} if clear is not None else current
    if clear is not None:
        for side in sides:
            act0 = _ARM0[side]
            lo, hi = _limits(model, act0)
            scratch.qpos[:] = data.qpos
            _set_arms(scratch, qadr, chosen)
            _sync(model, scratch)
            _refine(model, scratch, goal, side, qadr, vadr, lo, hi)
            chosen[side] = np.asarray(scratch.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
        refined_pen = _path_penetration(model, scratch, data, qadr, goal, current, chosen, sides, allow)
        if refined_pen > 1e-3:
            chosen = {side: q.copy() for side, q in clear.items()}
        else:
            _posed(model, scratch, data, qadr, goal, chosen, sides, allow, data.qpos[:2])
            clear_dist = max(_distance(model, scratch, goal, side) for side in sides)
    for side in sides:
        sol[_ARM0[side] : _ARM0[side] + 7] = chosen[side]
    goal["shift"] = None
    reachable = clear is not None and clear_dist <= TOUCH_M and kin_pen <= 1e-3
    if not reachable:
        goal["shift"] = _stance_for_raise(
            model, scratch, data, qadr, goal, current, kin_arms, sides, allow
        )
    gap = float(clear_dist) if clear is not None else float(max(kin_dist, TOUCH_M + 1.0))
    # One pixel is one hand. Forcing both onto it keeps the worse hand out of reach,
    # so the body stops short. Take the hand that actually gets closer.
    if goal.get("hand") == "both" and not reachable:
        best_gap = gap
        best_sol = sol
        best_side = None
        best_shift = goal.get("shift")
        for side in ("left", "right"):
            alt = dict(goal)
            alt["hand"] = side
            alt_sol, alt_gap = solve_arm(model, data, alt, qadr, vadr)
            if float(alt_gap) < best_gap:
                best_gap = float(alt_gap)
                best_sol = alt_sol
                best_side = side
                best_shift = alt.get("shift")
        if best_side is not None:
            goal["hand"] = best_side
            goal["shift"] = best_shift
            return best_sol, best_gap
    return sol, gap


_refine_scratch: dict[int, object] = {}


def refine_arm(model, data, goal: dict, qadr, vadr, solution: np.ndarray) -> tuple[np.ndarray, float]:
    """Move an existing arm pose onto the point from the body pose he has now.

    The grid solve is too slow to repeat while he sways. These are a few jacobian
    steps from the pose he already has, and a step that enters another body is dropped.
    """
    sol = np.asarray(solution, dtype=np.float64).copy()
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    sides = tuple(side for side in sides if side in _HAND_BODY)
    if not sides or goal.get("point") is None:
        return sol, float("inf")
    scratch = _refine_scratch.get(id(model))
    if scratch is None or int(scratch.qpos.shape[0]) != int(model.nq):
        scratch = mujoco.MjData(model)
        _refine_scratch[id(model)] = scratch
    scratch.qpos[:] = data.qpos
    for side in sides:
        act0 = _ARM0[side]
        lo, hi = _limits(model, act0)
        scratch.qpos[qadr[act0 : act0 + 7]] = sol[act0 : act0 + 7]
        _sync(model, scratch)
        _refine(model, scratch, goal, side, qadr, vadr, lo, hi)
        sol[act0 : act0 + 7] = np.asarray(scratch.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
    _set_arms(scratch, qadr, {side: sol[_ARM0[side] : _ARM0[side] + 7] for side in sides})
    _contacts(model, scratch)
    if _arm_penetration(model, scratch, sides, goal.get("body")) > 1e-3:
        sol = np.asarray(solution, dtype=np.float64).copy()
        scratch.qpos[:] = data.qpos
        _set_arms(scratch, qadr, {side: sol[_ARM0[side] : _ARM0[side] + 7] for side in sides})
        _sync(model, scratch)
    gap = max(_distance(model, scratch, goal, side) for side in sides)
    return sol, float(gap)


def servo_reach(model, data, cmd: np.ndarray, goal: dict | None, qadr, solution: np.ndarray | None) -> np.ndarray:
    """Pull the arm command toward a solved pose. The step is small so the body can balance."""
    if not goal or solution is None:
        return cmd
    out = np.array(cmd, dtype=np.float32, copy=True)
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    for side in sides:
        if side not in _ARM0:
            continue
        act0 = _ARM0[side]
        cmd0 = _CMD0[side]
        desired = np.asarray(solution[act0 : act0 + 7], dtype=np.float32)
        out[cmd0 : cmd0 + 7] = desired
    return out
