"""Move a hand toward a body landmark. Joint targets come from the Jacobian, not a pose table."""

from __future__ import annotations

import numpy as np
import mujoco

# Palm point in the wrist-yaw frame. +X runs out through the rubber hand.
_PALM = np.array([0.10, 0.0, 0.0], dtype=np.float64)
_HAND_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
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


def reach_goal_from_step(step: dict) -> dict | None:
    raw = step.get("hand_goal") if isinstance(step, dict) else None
    if not isinstance(raw, dict):
        nested = step.get("params") if isinstance(step, dict) else None
        raw = nested.get("hand_goal") if isinstance(nested, dict) else None
    if not isinstance(raw, dict):
        return None
    hand = str(raw.get("hand") or raw.get("body") or "").strip().lower()
    target = str(raw.get("target") or "").strip().lower()
    hand = {"left_hand": "left", "right_hand": "right", "лев": "left", "прав": "right"}.get(hand, hand)
    if hand in {"left", "right", "both"} and target == "head":
        return {"hand": hand, "target": "head"}
    return None


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


def _distance(model, scratch, side: str) -> float:
    point, _ = _palm(model, scratch, side)
    return float(np.linalg.norm(_head_target(model, scratch, side) - point))


def _refine(model, scratch, side: str, qadr, vadr, lo: np.ndarray, hi: np.ndarray) -> None:
    """Local steps after a candidate is already near the head."""
    act0 = _ARM0[side]
    for _ in range(20):
        point, bid = _palm(model, scratch, side)
        err = _head_target(model, scratch, side) - point
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


def solve_arm(model, data, goal: dict, qadr, vadr) -> np.ndarray:
    """Grid the shoulder and elbow, keep the pose whose palm is closest to the target."""
    sol = np.asarray(data.qpos[qadr], dtype=np.float64).copy()
    scratch = mujoco.MjData(model)
    sides = ("left", "right") if goal.get("hand") == "both" else (str(goal.get("hand")),)
    for side in sides:
        if side not in _HAND_BODY:
            continue
        act0 = _ARM0[side]
        lo, hi = _limits(model, act0)
        current = np.asarray(data.qpos[qadr[act0 : act0 + 7]], dtype=np.float64).copy()
        best_q = current.copy()
        best_cost = _distance(model, data, side)
        for pitch in np.linspace(lo[0], hi[0], 7):
            for roll in np.linspace(lo[1], hi[1], 5):
                for elbow in np.linspace(lo[3], hi[3], 5):
                    trial = current.copy()
                    trial[0], trial[1], trial[3] = pitch, roll, elbow
                    scratch.qpos[:] = data.qpos
                    scratch.qpos[qadr[act0 : act0 + 7]] = trial
                    _sync(model, scratch)
                    dist = _distance(model, scratch, side)
                    cost = dist + 0.03 * float(np.linalg.norm(trial - current))
                    if cost < best_cost:
                        best_cost = cost
                        best_q = trial.copy()
        scratch.qpos[:] = data.qpos
        scratch.qpos[qadr[act0 : act0 + 7]] = best_q
        _sync(model, scratch)
        _refine(model, scratch, side, qadr, vadr, lo, hi)
        sol[act0 : act0 + 7] = scratch.qpos[qadr[act0 : act0 + 7]]
    return sol


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
