"""Move a hand toward a body landmark. Joint targets come from the Jacobian, not a pose table."""

from __future__ import annotations

import numpy as np
import mujoco

# Palm point in the wrist-yaw frame. +X runs out through the rubber hand.
# Knuckle of the Dex3 palm, in the wrist-yaw frame. +X runs out through the fingers.
_PALM = np.array([0.14, 0.0, 0.0], dtype=np.float64)
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
# Aim this far in front of the surface so the curl meets the object.
STANDOFF_M = 0.03
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
        err = _target_point(model, scratch, goal, side) - point
        if float(np.linalg.norm(err)) < 0.04:
            return
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, scratch, jacp, jacr, point, bid)
        cols = np.asarray(vadr[act0 : act0 + 7], dtype=np.int32)
        jac = jacp[:, cols]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.08 * np.eye(3), err)
        q = _clip_arm(scratch.qpos[qadr[act0 : act0 + 7]] + np.clip(dq, -0.15, 0.15), lo, hi)
        scratch.qpos[qadr[act0 : act0 + 7]] = q
        _sync(model, scratch)


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
    """Grid the shoulder and elbow. Keep a pose only when the arm does not enter another body."""
    sol = np.asarray(data.qpos[qadr], dtype=np.float64).copy()
    scratch = mujoco.MjData(model)
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    sides = tuple(side for side in sides if side in _HAND_BODY)
    allow = goal.get("body")
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
        kinematic[side] = best_q
        by_roll[side] = per_roll
        sol[act0 : act0 + 7] = q_now
    if not sides:
        goal["shift"] = None
        return sol, float("inf")

    # One candidate per shoulder roll, so a swept-back shoulder is scored even if a straight arm is closer.
    candidates = []
    roll_ids = range(5)
    for ri in roll_ids:
        arms = {}
        dist = 0.0
        ok = True
        for side in sides:
            picked = by_roll[side].get(ri)
            if picked is None:
                ok = False
                break
            dist = max(dist, float(picked[0]))
            arms[side] = picked[1]
        if ok:
            candidates.append((dist, arms))
    clear = None
    clear_dist = float("inf")
    for dist, arms in candidates:
        pen = _path_penetration(model, scratch, data, qadr, goal, current, arms, sides, allow)
        if pen <= 1e-3 and dist < clear_dist:
            clear = {side: q.copy() for side, q in arms.items()}
            clear_dist = dist
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
    return sol, gap


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
