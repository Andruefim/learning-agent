"""System 2 Deliberative Executor: Real RL Walking Approach, Raycast Grounding, Whole-Body IK & Primitives."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from PIL import Image

from agent.config import LOOK_H, LOOK_W, SLEEP_HORIZON, SLEEP_IMG_SIZE, SUBSTEPS_PER_COG_STEP
from agent.h2 import N_ACT, SQUAT_Q, STAND_Q
from agent.planner import VisionLanguagePlanner
from agent.system0 import (
    MAT_HORIZ_HOOK,
    MAT_PUSH_FRONT,
    MAT_VERT_HOOK,
    System0Controller,
    default_raw_action,
    extract_proprio,
    heading_z,
    raw_to_norm_action,
    solve_whole_body_ik,
)

# Robot-local primitive directions (+X = forward, +Y = left, +Z = up)
PRIM_LOCAL_DIR = {
    "press_down": np.array([0.05, 0.0, -1.0], dtype=np.float64),
    "push_forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "pull_back": np.array([-1.0, 0.0, -0.22], dtype=np.float64),  # back + slight downward arc for hinges
    "lift_up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "slide_left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "slide_right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}
PRIM_GRIP = {"pull_back": 1.0, "lift_up": 0.65}


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _subtree_bodies(model: mujoco.MjModel, root_body: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for b in range(int(model.nbody)):
        p = int(model.body_parentid[b])
        if b != p:
            kids.setdefault(p, []).append(b)
    seen = {int(root_body)}
    stack = [int(root_body)]
    while stack:
        cur = stack.pop()
        for ch in kids.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return seen


def _resize_rgb(rgb: np.ndarray, size: int = SLEEP_IMG_SIZE) -> np.ndarray:
    return np.array(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.uint8, copy=True)


class System2Executor:
    """Deliberative controller bridging VLM plans with free-standing MuJoCo physics."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sys0: System0Controller,
        planner: VisionLanguagePlanner,
        render_head_fn: Callable[[], np.ndarray],
        step_cog_fn: Callable[[np.ndarray, bool, bool], None],
        is_aborted_fn: Callable[[], bool],
        log_fn: Callable[[str, str], None],
        sim_lock: Optional[threading.RLock] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.scratch = mujoco.MjData(model)
        self.stance_probe = mujoco.MjData(model)
        self._sim_lock = sim_lock or threading.RLock()
        self.sys0 = sys0
        self.planner = planner
        self._render_head = render_head_fn
        self._step_cog = step_cog_fn   # step_cog_fn(raw_action, tuck_left, tuck_right)
        self._aborted = is_aborted_fn
        self._log = log_fn
        self.cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head"))
        self.robot_bodies = _subtree_bodies(model, sys0.pelvis_id)
        self.marker_pos: Optional[np.ndarray] = None

    def uv_to_world(self, u: float, v: float) -> Optional[Tuple[np.ndarray, str]]:
        """Project head-camera pixel (u, v) into 3D world point via MuJoCo raycast."""
        w, h = LOOK_W, LOOK_H
        fovy = float(self.model.cam_fovy[self.cam_id])
        f = 0.5 * h / math.tan(math.radians(fovy) * 0.5)
        ray_cam = np.array([(u - w * 0.5) / f, -(v - h * 0.5) / f, -1.0], dtype=np.float64)
        ray_cam /= max(float(np.linalg.norm(ray_cam)), 1e-8)

        with self._sim_lock:
            origin = np.asarray(self.data.cam_xpos[self.cam_id], dtype=np.float64).copy()
            rot = np.asarray(self.data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3).copy()
            dir_w = rot @ ray_cam
            dir_w /= max(float(np.linalg.norm(dir_w)), 1e-8)

            geomid = np.zeros(1, dtype=np.int32)
            groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
            dist = float(mujoco.mj_ray(self.model, self.data, origin + 0.08 * dir_w, dir_w, groups, 1, self.sys0.torso_id, geomid))
        if dist < 0.0 or geomid[0] < 0:
            return None
        hit_dist = 0.08 + dist
        if hit_dist > 3.2:
            return None
        body_id = int(self.model.geom_bodyid[int(geomid[0])])
        if body_id in self.robot_bodies:
            return None
        bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
        name = f"{bname}/{gname}".lower()
        pt = origin + hit_dist * dir_w
        if "floor" in name or "wall" in name or pt[2] < 0.08:
            return None
        return pt, bname

    def _current_proprio(self, grip: float = 0.0) -> np.ndarray:
        with self._sim_lock:
            return extract_proprio(
                self.data, self.sys0.pelvis_id, self.sys0.torso_id, self.sys0.wrist_r_id, self.sys0.qadr, grip
            )

    def _run_step(
        self,
        raw: np.ndarray,
        traj: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        *,
        tuck_left: bool = True,
        tuck_right: bool = False,
    ) -> bool:
        with self._sim_lock:
            fell = self.sys0.fell()
        if self._aborted() or fell:
            return False
        if traj is not None:
            rgb = self._render_head()
            pr = self._current_proprio(float(raw[19]))
            traj.append((_resize_rgb(rgb), pr, raw_to_norm_action(raw)))
        self._step_cog(raw, tuck_left, tuck_right)
        with self._sim_lock:
            fell = self.sys0.fell()
        return not (self._aborted() or fell)

    def walk_to(self, target_xy: np.ndarray, target_yaw: float, max_steps: int = 80) -> bool:
        """Closed-loop locomotion using G1LegPolicy (no teleport!)."""
        tgt = np.asarray(target_xy, dtype=np.float64)
        for _ in range(max_steps):
            with self._sim_lock:
                fell = self.sys0.fell()
                pelvis = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
                yaw = heading_z(self.data.qpos[3:7])
            if self._aborted() or fell:
                return False
            dvec = tgt - pelvis
            dist = float(np.linalg.norm(dvec))
            if dist < 0.14 and abs(_wrap_angle(target_yaw - yaw)) < 0.18:
                break

            raw = default_raw_action()
            if dist >= 0.14:
                desired_yaw = math.atan2(dvec[1], dvec[0])
                herr = _wrap_angle(desired_yaw - yaw)
                # Walk backward if close and slightly behind, otherwise turn toward target
                if dist < 0.38 and abs(herr) > 2.2:
                    raw[0] = -0.22
                    raw[1] = float(np.clip(0.8 * _wrap_angle(target_yaw - yaw), -0.5, 0.5))
                elif abs(herr) > 0.45:
                    raw[0] = 0.0
                    raw[1] = float(np.clip(1.3 * herr, -0.65, 0.65))
                else:
                    raw[0] = float(np.clip(0.75 * dist, 0.16, 0.40))
                    raw[1] = float(np.clip(1.2 * herr, -0.55, 0.55))
            else:
                herr = _wrap_angle(target_yaw - yaw)
                raw[0] = 0.0
                raw[1] = float(np.clip(1.4 * herr, -0.60, 0.60))

            self._run_step(raw, None, tuck_left=True, tuck_right=True)

        # Fade RL gait smoothly onto stiff stand PD (~1.1 s)
        stand = default_raw_action()
        for _ in range(11):
            if not self._run_step(stand, None, tuck_left=True, tuck_right=True):
                return False
        return True

    def _choose_stance(self, pt: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Select collision-free standing pose (xy, yaw, squat) maximizing IK reachability for `pt`."""
        with self._sim_lock:
            pelvis0 = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
            self.stance_probe.qpos[:] = self.data.qpos
            self.stance_probe.qvel[:] = self.data.qvel
        to_p = pt[:2] - pelvis0
        yaw_face = math.atan2(to_p[1], to_p[0]) if float(np.linalg.norm(to_p)) > 0.15 else 0.0
        # Snap cardinal kitchen face (+X facing counter) if already roughly facing +X
        if abs(_wrap_angle(yaw_face)) < 0.75 and pt[0] > 0.55:
            yaw_face = 0.0

        fwd = np.array([math.cos(yaw_face), math.sin(yaw_face)], dtype=np.float64)
        left = np.array([-fwd[1], fwd[0]], dtype=np.float64)

        best = (1e9, pelvis0, yaw_face, 0.05)
        for d_front in (0.40, 0.48, 0.56):
            for d_left in (0.14, 0.22, 0.06):
                for sq in (0.05, 0.25, 0.45):
                    xy = pt[:2] - d_front * fwd + d_left * left
                    self.stance_probe.qpos[0:2] = xy
                    self.stance_probe.qpos[3:7] = [math.cos(0.5 * yaw_face), 0.0, 0.0, math.sin(0.5 * yaw_face)]
                    _, err = solve_whole_body_ik(
                        self.model,
                        self.stance_probe,
                        self.scratch,
                        self.sys0.qadr,
                        self.sys0.vadr,
                        self.sys0.wrist_r_id,
                        pt,
                        MAT_HORIZ_HOOK,
                        squat=sq,
                    )
                    if err < best[0]:
                        best = (err, xy, yaw_face, sq)
        return best[1], best[2], best[3], best[0]

    def _move_hook(
        self,
        tgt_pos: np.ndarray,
        tgt_mat: np.ndarray,
        squat: float,
        grip: float,
        raw_cur: np.ndarray,
        steps: int,
        traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, float]:
        q_init = np.zeros(N_ACT, dtype=np.float32)
        q_init[12] = float(raw_cur[3])
        q_init[14] = float(raw_cur[4])
        q_init[22:29] = raw_cur[5:12]
        q_init[15:22] = raw_cur[12:19]

        with self._sim_lock:
            self.stance_probe.qpos[:] = self.data.qpos
        q_sol, _ = solve_whole_body_ik(
            self.model,
            self.stance_probe,
            self.scratch,
            self.sys0.qadr,
            self.sys0.vadr,
            self.sys0.wrist_r_id,
            tgt_pos,
            tgt_mat,
            squat=squat,
            q_init=q_init,
        )
        tgt_raw = default_raw_action()
        tgt_raw[2] = float(squat)
        tgt_raw[3] = float(q_sol[12])
        tgt_raw[4] = float(q_sol[14])
        tgt_raw[5:12] = q_sol[22:29]
        tgt_raw[19] = float(grip)

        r0 = raw_cur.copy()
        cur = r0
        for k in range(1, steps + 1):
            frac = k / steps
            cur = (r0 + (tgt_raw - r0) * frac).astype(np.float32)
            cur[0:2] = 0.0
            if not self._run_step(cur, traj, tuck_left=True, tuck_right=False):
                break
        with self._sim_lock:
            err = float(np.linalg.norm(self.sys0.hook_pos("right") - tgt_pos))
        return cur, err

    def _try_primitive(
        self, prim: str, p: np.ndarray, squat: float, command: str
    ) -> Tuple[bool, str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        with self._sim_lock:
            yaw = heading_z(self.data.qpos[3:7])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        d_local = PRIM_LOCAL_DIR.get(prim, PRIM_LOCAL_DIR["pull_back"])
        d_world = Rz @ d_local
        d_world /= max(float(np.linalg.norm(d_world)), 1e-8)

        if prim in ("slide_left", "slide_right"):
            R = Rz @ MAT_VERT_HOOK
        elif prim == "push_forward":
            R = Rz @ MAT_PUSH_FRONT
        else:
            R = Rz @ MAT_HORIZ_HOOK

        before_rgb = self._render_head().copy()
        traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        raw = default_raw_action()
        raw[2] = float(squat)

        # 1. Pre-contact cleanly outside target surface (robot-facing side)
        fwd_w = Rz @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if prim == "pull_back":
            pre = p - 0.10 * fwd_w + np.array([0.0, 0.0, 0.06])
            raw, _ = self._move_hook(pre, R, squat, 0.0, raw, steps=8, traj=traj)
            over = p + 0.015 * fwd_w + np.array([0.0, 0.0, 0.03])
            raw, _ = self._move_hook(over, R, squat, 0.0, raw, steps=6, traj=traj)
            contact = p + 0.015 * fwd_w - np.array([0.0, 0.0, 0.01])
            raw, _ = self._move_hook(contact, R, squat, 0.0, raw, steps=5, traj=traj)
        else:
            pre = p - 0.08 * d_world + np.array([0.0, 0.0, 0.02])
            raw, _ = self._move_hook(pre, R, squat, 0.0, raw, steps=8, traj=traj)
            raw, _ = self._move_hook(p, R, squat, 0.0, raw, steps=6, traj=traj)

        # 3. Engage fingers firmly if hooking/lifting
        grip = PRIM_GRIP.get(prim, 0.0)
        raw[19] = grip
        for _ in range(6 if grip > 0.0 else 2):
            if not self._run_step(raw, traj, tuck_left=True, tuck_right=False):
                return False, "потеряно равновесие или остановлено", traj

        # 4. Execute primitive stroke (allow physical contact lag up to 0.24 m)
        travel = 0.28 if prim == "pull_back" else (0.22 if prim == "push_forward" else 0.18)
        n_seg = 8 if prim == "pull_back" else 6
        for k in range(1, n_seg + 1):
            tgt = p + d_world * (travel * k / n_seg)
            raw, err = self._move_hook(tgt, R, squat, grip, raw, steps=2, traj=traj)
            with self._sim_lock:
                fell = self.sys0.fell()
            if err > 0.24 or self._aborted() or fell:
                break

        mid_rgb = self._render_head().copy()

        # 5. Release and retract back toward robot cleanly before returning to stand
        raw[19] = 0.0
        with self._sim_lock:
            cur_hook = self.sys0.hook_pos("right").copy()
        retract = cur_hook - 0.09 * fwd_w + np.array([0.0, 0.0, 0.05])
        raw, _ = self._move_hook(retract, R, 0.05, 0.0, raw, steps=5, traj=None)
        stand = default_raw_action()
        for k in range(1, 8):
            frac = k / 7.0
            blend = (raw + (stand - raw) * frac).astype(np.float32)
            self._run_step(blend, None, tuck_left=True, tuck_right=(k >= 5))

        after_rgb = self._render_head().copy()
        verdict = self.planner.judge(before_rgb, mid_rgb, command)
        if not verdict["done"]:
            verdict2 = self.planner.judge(before_rgb, after_rgb, command)
            if verdict2["done"]:
                verdict = verdict2
        return bool(verdict["done"]), str(verdict.get("what") or ""), traj

    def _run_body_skill(
        self, plan: Dict[str, Any], command: str
    ) -> Tuple[bool, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        skill = plan.get("skill", "stand")
        params = plan.get("params") or {}
        traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        raw = default_raw_action()

        direction = str(params.get("direction") or "").lower()
        hand = str(params.get("hand") or "both").lower()
        pose = str(params.get("pose") or "").lower()

        if skill == "locomote":
            vx = -0.30 if "back" in direction else 0.38
            steps_hint = float(params.get("steps") or 3)
            dur_ticks = int(np.clip(steps_hint * 6, 12, 40))
            raw[0] = vx
            for _ in range(dur_ticks):
                if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            stand = default_raw_action()
            for _ in range(10):
                if not self._run_step(stand, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            return True, traj

        if skill == "turn":
            wz = -0.55 if "right" in direction else 0.55
            angle_deg = float(params.get("angle") or 60)
            dur_ticks = int(np.clip(abs(angle_deg) / 6.0, 8, 30))
            raw[1] = wz
            for _ in range(dur_ticks):
                if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            stand = default_raw_action()
            for _ in range(10):
                if not self._run_step(stand, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            return True, traj

        if skill == "squat":
            raw[2] = 0.55
            for _ in range(25):
                if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            return True, traj

        if skill in ("reach", "wave"):
            tuck_l = hand == "right"
            tuck_r = hand == "left"
            if pose == "down":
                raw = default_raw_action()
            elif pose == "t":
                raw[5:12] = np.array([0.0, -1.25, 0.0, 0.35, 0.0, 0.0, 0.0], dtype=np.float32)
                raw[12:19] = np.array([0.0, 1.25, 0.0, 0.35, 0.0, 0.0, 0.0], dtype=np.float32)
                tuck_l = tuck_r = False
            else:
                raise_arm = np.array([-1.55, -0.15, 0.0, 0.70, 0.0, 0.0, 0.0], dtype=np.float32)
                raise_arm_l = np.array([-1.55, 0.15, 0.0, 0.70, 0.0, 0.0, 0.0], dtype=np.float32)
                if hand in ("right", "both"):
                    raw[5:12] = raise_arm
                if hand in ("left", "both"):
                    raw[12:19] = raise_arm_l
            for _ in range(22):
                if not self._run_step(raw, traj, tuck_left=tuck_l, tuck_right=tuck_r):
                    return False, traj
            return True, traj

        # Default: steady standing hold
        for _ in range(10):
            if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                return False, traj
        return True, traj

    def execute(self, command: str, *, lessons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Full System 2 execution pipeline returning success status, Wake chunks, and failure diagnostics."""
        self.marker_pos = None
        with self._sim_lock:
            fell0 = self.sys0.fell()
        if fell0:
            return {"ok": False, "reason": "робот лежит", "samples": []}
        rgb0 = self._render_head()
        plan = self.planner.plan_command(rgb0, command, lessons=lessons)
        if not plan.get("ok"):
            return {"ok": False, "reason": "VLM недоступна", "samples": []}

        if plan["mode"] == "body_skill":
            self._log("sys2", f"[System 2] Базовый навык: {plan.get('skill')} {plan.get('params')}")
            ok, traj = self._run_body_skill(plan, command)
            samples = self._build_chunks(traj, command)
            if ok:
                return {"ok": True, "mode": "body_skill", "reason": "", "samples": samples}
            with self._sim_lock:
                fell_now = self.sys0.fell()
            after_rgb = self._render_head()
            return {
                "ok": False,
                "mode": "body_skill",
                "reason": "потеряно равновесие или остановлено",
                "samples": samples,
                "failure_info": {
                    "before_rgb": rgb0,
                    "after_rgb": after_rgb,
                    "tried_motions": [str(plan.get("skill") or "body_skill")],
                    "ik_err_cm": 0.0,
                    "fell": fell_now,
                    "what": "потеря равновесия при выполнении движения корпуса",
                },
            }

        # Object manipulation: search with head rotation if not visible in initial view
        found = plan if plan.get("visible") else None
        if found is None:
            wz = 0.72 if plan.get("search_dir") == "left" else -0.72
            self._log("sys2", f"[System 2] Цель не в кадре, осматриваюсь ({plan.get('search_dir')})...")
            for _ in range(9):
                turn_raw = default_raw_action()
                turn_raw[1] = wz
                for _ in range(10):
                    if not self._run_step(turn_raw, None, tuck_left=True, tuck_right=True):
                        return {"ok": False, "reason": "Остановлено", "samples": []}
                g = self.planner.ground(self._render_head(), command, lessons=lessons)
                if g.get("visible"):
                    found = g
                    break

        if found is None or not found.get("uv"):
            self._log("warn", f"[System 2] Цель «{command}» не найдена камерой.")
            return {
                "ok": False,
                "reason": "not_visible",
                "samples": [],
                "failure_info": {
                    "before_rgb": rgb0,
                    "after_rgb": self._render_head(),
                    "tried_motions": ["search"],
                    "ik_err_cm": 0.0,
                    "fell": False,
                    "what": "целевой объект не обнаружен в поле зрения камеры",
                },
            }

        hit = self.uv_to_world(*found["uv"])
        if hit is None:
            self._log("warn", "[System 2] Луч камеры не попал на манипулируемый объект.")
            return {
                "ok": False,
                "reason": "raycast_miss",
                "samples": [],
                "failure_info": {
                    "before_rgb": rgb0,
                    "after_rgb": self._render_head(),
                    "tried_motions": ["raycast"],
                    "ik_err_cm": 0.0,
                    "fell": False,
                    "what": "точка на изображении не проецируется на объект (попала в фон/стену)",
                },
            }

        pt, bname = hit
        self.marker_pos = pt.copy()
        self._log("sys2", f"[Grounding] {found.get('object') or bname}/{found.get('part') or 'handle'} -> {np.round(pt, 2)}; шагаю к цели...")

        stance_xy, stance_yaw, squat, ik_err = self._choose_stance(pt)
        if not self.walk_to(stance_xy, stance_yaw):
            with self._sim_lock:
                fell_now = self.sys0.fell()
            return {
                "ok": False,
                "reason": "walk_aborted",
                "samples": [],
                "failure_info": {
                    "before_rgb": rgb0,
                    "after_rgb": self._render_head(),
                    "target_pt": pt.copy(),
                    "tried_motions": ["walk_to"],
                    "ik_err_cm": ik_err * 100.0,
                    "fell": fell_now,
                    "what": "не удалось подойти к выбранной точке",
                },
            }

        # Re-ground closer up once standing stably at target stance
        g2 = self.planner.ground(self._render_head(), command, lessons=lessons)
        motions = found["motions"]
        if g2.get("visible") and g2.get("uv"):
            hit2 = self.uv_to_world(*g2["uv"])
            if hit2 is not None and float(np.linalg.norm(hit2[0] - pt)) < 0.35:
                pt = hit2[0]
                self.marker_pos = pt.copy()
                _, _, squat, ik_err = self._choose_stance(pt)
                motions = g2["motions"]
                self._log("sys2", f"[Grounding вблизи] {np.round(pt, 2)} · движения: {motions[:3]}")

        tried: list[str] = []
        best_traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        last_what = "attempts_exhausted"
        close_rgb = self._render_head()

        for prim in motions[:3]:
            with self._sim_lock:
                fell = self.sys0.fell()
            if self._aborted() or fell:
                if fell:
                    last_what = "потеря равновесия"
                break
            tried.append(prim)
            self._log("sys2", f"[Примитив] Пробую `{prim}` (присед={squat:.2f}, IK err={ik_err*100:.1f} см)")
            done, what, traj = self._try_primitive(prim, pt, squat, command)
            if len(traj) > len(best_traj):
                best_traj = traj
            if done:
                self.marker_pos = None
                self._log("info", f"[VLM Вердикт] Успех ({prim}): {what}")
                return {"ok": True, "mode": "manipulate", "primitive": prim, "samples": self._build_chunks(traj, command)}
            last_what = what or last_what
            self._log("warn", f"[Примитив] `{prim}` не завершил задачу ({what})")

        self.marker_pos = None
        with self._sim_lock:
            fell_end = self.sys0.fell()
        after_rgb = self._render_head()
        return {
            "ok": False,
            "mode": "manipulate",
            "reason": "attempts_exhausted",
            "samples": self._build_chunks(best_traj, command),
            "failure_info": {
                "before_rgb": close_rgb,
                "after_rgb": after_rgb,
                "target_pt": pt.copy(),
                "tried_motions": tried,
                "ik_err_cm": ik_err * 100.0,
                "fell": fell_end,
                "what": last_what,
            },
        }

    @staticmethod
    def _build_chunks(
        traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], command: str, horizon: int = SLEEP_HORIZON
    ) -> List[Tuple[np.ndarray, np.ndarray, str, np.ndarray]]:
        """Slice real future H-step action trajectories [(rgb_128, proprio, command, chunk_H_A)]."""
        n = len(traj)
        if n == 0:
            return []
        acts = [item[2] for item in traj]
        out = []
        for t in range(n):
            chunk = [acts[min(t + k, n - 1)] for k in range(horizon)]
            out.append((traj[t][0], traj[t][1], command, np.stack(chunk, axis=0).astype(np.float32)))
        return out
