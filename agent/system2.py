"""System 2 Deliberative Executor: Bumper-Safe RL Walk, Mental Scratch Rehearsal, Whole-Body IK & Primitives."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from PIL import Image

from agent.config import LOOK_H, LOOK_W, SLEEP_HORIZON, SLEEP_IMG_SIZE
from agent.h2 import N_ACT
from agent.planner import VisionLanguagePlanner
from agent.system0 import (
    MAT_HORIZ_HOOK,
    MAT_PUSH_FRONT,
    MAT_VERT_HOOK,
    System0Controller,
    com_err_xy,
    default_raw_action,
    extract_proprio,
    heading_z,
    raw_to_norm_action,
    solve_whole_body_ik,
)

# Robot-local primitive directions (+X = forward, +Y = left, +Z = up)
PRIM_LOCAL_DIR = {
    "press_down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "push_forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "pull_back": np.array([-1.0, 0.0, -0.22], dtype=np.float64),  # back + slight downward arc for hinges
    "lift_up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "slide_left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "slide_right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}
PRIM_GRIP = {"pull_back": 1.0, "lift_up": 0.65, "press_down": 0.35}


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
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.copy()
    return np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8, copy=True)


class System2Executor:
    """Deliberative controller pairing safe locomotion, MuJoCo scratch rehearsal, whole-body IK, and VLM checks."""

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
        **_unused_callbacks: Any,
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

    def reset_memory(self) -> None:
        self.marker_pos = None

    def _ray_cast_env(
        self, sim_data: mujoco.MjData, origin: np.ndarray, dir_w: np.ndarray, *, max_dist: float = 3.2
    ) -> Optional[Tuple[float, np.ndarray, str]]:
        groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        geomid = np.zeros(1, dtype=np.int32)
        dist = float(mujoco.mj_ray(self.model, sim_data, origin, dir_w, groups, 1, self.sys0.torso_id, geomid))
        if dist < 0.0 or geomid[0] < 0 or dist > max_dist:
            return None
        body_id = int(self.model.geom_bodyid[int(geomid[0])])
        if body_id in self.robot_bodies:
            return None
        bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
        name = f"{bname}/{gname}".lower()
        pt = origin + dist * dir_w
        return dist, pt, name

    def uv_to_world(self, u: float, v: float) -> Optional[Tuple[np.ndarray, str]]:
        """Project head-camera pixel (u, v) into 3D world point with a local front-surface guard."""
        w, h = LOOK_W, LOOK_H
        fovy = float(self.model.cam_fovy[self.cam_id])
        f = 0.5 * h / math.tan(math.radians(fovy) * 0.5)

        def _dir_for(px: float, py: float, rot: np.ndarray) -> np.ndarray:
            ray_cam = np.array([(px - w * 0.5) / f, -(py - h * 0.5) / f, -1.0], dtype=np.float64)
            ray_cam /= max(float(np.linalg.norm(ray_cam)), 1e-8)
            dw = rot @ ray_cam
            dw /= max(float(np.linalg.norm(dw)), 1e-8)
            return dw

        with self._sim_lock:
            origin = np.asarray(self.data.cam_xpos[self.cam_id], dtype=np.float64).copy()
            rot = np.asarray(self.data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3).copy()
            center_dir = _dir_for(u, v, rot)
            center_hit = self._ray_cast_env(self.data, origin + 0.08 * center_dir, center_dir, max_dist=3.12)

            # Cast 3x3 ring around (u, v) so misclicks inside oven/microwave glass snap to the front surface
            front_dists: list[float] = []
            for dy in (-14.0, 0.0, 14.0):
                for dx in (-14.0, 0.0, 14.0):
                    d_w = _dir_for(np.clip(u + dx, 1.0, w - 2.0), np.clip(v + dy, 1.0, h - 2.0), rot)
                    hit = self._ray_cast_env(self.data, origin + 0.08 * d_w, d_w, max_dist=3.12)
                    if hit is not None:
                        dist, pt, name = hit
                        if "floor" not in name and "wall" not in name and pt[2] >= 0.08:
                            front_dists.append(0.08 + dist)

        if center_hit is None:
            return None
        c_dist_raw, pt, name = center_hit
        c_dist = 0.08 + c_dist_raw
        if "floor" in name or "wall" in name or pt[2] < 0.08:
            return None

        if front_dists:
            front_min = float(np.percentile(front_dists, 20))
            if c_dist > front_min + 0.14:
                pt = origin + (front_min + 0.015) * center_dir

        bname = name.split("/", 1)[0]
        return pt, bname

    def _front_bumper_clearance(self, sim_data: Optional[mujoco.MjData] = None) -> float:
        """Minimum horizontal distance to obstacles across floor-prop/shin/hip/countertop heights ahead."""
        min_d = 3.0
        with self._sim_lock:
            sd = sim_data or self.data
            px, py = float(sd.qpos[0]), float(sd.qpos[1])
            yaw = heading_z(sd.qpos[3:7])
            heights = (0.15, 0.34, 0.62, 0.86)
            for ang in (-0.45, -0.20, 0.0, 0.20, 0.45):
                th = yaw + ang
                dir_w = np.array([math.cos(th), math.sin(th), 0.0], dtype=np.float64)
                for hz in heights:
                    off = 0.11 if hz >= 0.60 else 0.15
                    orig = np.array([px + off * dir_w[0], py + off * dir_w[1], hz], dtype=np.float64)
                    hit = self._ray_cast_env(sd, orig, dir_w, max_dist=2.8)
                    if hit is not None:
                        d = off + hit[0]
                        if d < min_d:
                            min_d = d
        return float(min_d)

    def _stance_floor_clearance(self, sim_data: mujoco.MjData) -> float:
        """Closest obstacle/floor-prop around all horizontal sides (catches knife block near stance feet)."""
        min_d = 3.0
        with self._sim_lock:
            px, py = float(sim_data.qpos[0]), float(sim_data.qpos[1])
            yaw = heading_z(sim_data.qpos[3:7])
            for ang in (-1.6, -1.0, -0.5, 0.0, 0.5, 1.0, 1.6):
                th = yaw + ang
                dir_w = np.array([math.cos(th), math.sin(th), 0.0], dtype=np.float64)
                for hz in (0.15, 0.34):
                    orig = np.array([px + 0.16 * dir_w[0], py + 0.16 * dir_w[1], hz], dtype=np.float64)
                    hit = self._ray_cast_env(sim_data, orig, dir_w, max_dist=1.0)
                    if hit is not None:
                        d = 0.16 + hit[0]
                        if d < min_d:
                            min_d = d
        return float(min_d)

    def _forward_clearance(self) -> float:
        return self._front_bumper_clearance()

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
            unstable = self.sys0.fell() or self.sys0.recovering
        if self._aborted() or unstable:
            return False
        if traj is not None:
            rgb = self._render_head()
            pr = self._current_proprio(float(raw[19]))
            traj.append((_resize_rgb(rgb), pr, raw_to_norm_action(raw)))
        self._step_cog(raw, tuck_left, tuck_right)
        with self._sim_lock:
            unstable = self.sys0.fell() or self.sys0.recovering
        return not (self._aborted() or unstable)

    def _step_slew(
        self,
        prev_vx: float,
        prev_wz: float,
        tgt_vx: float,
        tgt_wz: float,
        *,
        max_dvx: float = 0.08,
        max_dwz: float = 0.12,
        traj: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[bool, float, float]:
        """Advance one slew-rate-limited locomotion tick so G1LegPolicy never sees abrupt impulses."""
        vx = float(np.clip(tgt_vx, prev_vx - max_dvx, prev_vx + max_dvx))
        wz = float(np.clip(tgt_wz, prev_wz - max_dwz, prev_wz + max_dwz))
        raw = default_raw_action()
        raw[0] = vx
        raw[1] = wz
        ok = self._run_step(raw, traj, tuck_left=True, tuck_right=True)
        return ok, vx, wz

    def _ramp_to_stop(
        self,
        prev_vx: float,
        prev_wz: float,
        *,
        wait_hold: bool = True,
        max_hold: int = 22,
        traj: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> bool:
        """Ramp residual locomotion velocities smoothly to zero and optionally wait for gait->stand fade."""
        vx, wz = float(prev_vx), float(prev_wz)
        for _ in range(5):
            if abs(vx) < 0.04 and abs(wz) < 0.04:
                break
            ok, vx, wz = self._step_slew(vx, wz, 0.0, 0.0, traj=traj)
            if not ok:
                return False
        return self._wait_gait_hold(traj, max_ticks=max_hold) if wait_hold else True

    def _safe_back_up(self, *, target_clear: float = 0.40, max_ticks: int = 16) -> bool:
        """Back away smoothly into open floor until front bumper clearance reaches target_clear."""
        if self._front_bumper_clearance() >= target_clear:
            return True
        vx, wz = 0.0, 0.0
        for _ in range(max_ticks):
            if self._front_bumper_clearance() >= target_clear:
                break
            ok, vx, wz = self._step_slew(vx, wz, -0.20, 0.0, max_dvx=0.07)
            if not ok:
                return False
        return self._ramp_to_stop(vx, wz, wait_hold=True, max_hold=18)

    def _wait_gait_hold(
        self,
        traj: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        max_ticks: int = 24,
        speed_tol: float = 0.05,
        min_ticks: int = 10,
    ) -> bool:
        """Let G1LegPolicy fade smoothly to 100% stiff stand PD (walk.holding == True) before upper-body moves."""
        stand = default_raw_action()
        for k in range(max_ticks):
            if not self._run_step(stand, traj, tuck_left=True, tuck_right=True):
                return False
            with self._sim_lock:
                holding = self.sys0.walk is None or self.sys0.walk.holding
                speed = float(np.linalg.norm(self.data.qvel[0:2]))
            if k >= min_ticks and holding and speed < speed_tol:
                return True
        return True

    def _walk_direct(
        self,
        tgt: np.ndarray,
        target_yaw: float,
        *,
        max_steps: int = 125,
        stop_dist: float = 0.08,
        settle: bool = True,
    ) -> bool:
        pos_done = False
        prev_vx = 0.0
        prev_wz = 0.0
        for _ in range(max_steps):
            with self._sim_lock:
                fell = self.sys0.fell()
                pelvis = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
                yaw = heading_z(self.data.qpos[3:7])
            if self._aborted() or fell:
                return False
            dvec = tgt - pelvis
            dist = float(np.linalg.norm(dvec))
            yaw_err = abs(_wrap_angle(target_yaw - yaw))
            fwd_t = np.array([math.cos(target_yaw), math.sin(target_yaw)], dtype=np.float64)
            past_tgt_plane = settle and dist < 0.36 and float(np.dot(pelvis - tgt, fwd_t)) >= 0.02
            if dist < stop_dist or past_tgt_plane:
                pos_done = True
            elif settle and pos_done and dist > 0.32 and not past_tgt_plane:
                pos_done = False

            if (pos_done or dist < stop_dist) and ((not settle) or yaw_err < 0.12):
                break

            bumper = self._front_bumper_clearance()
            raw = default_raw_action()
            if (not pos_done) and dist >= stop_dist:
                vec_yaw = math.atan2(dvec[1], dvec[0])
                desired_yaw = vec_yaw
                if settle and dist < 0.24:
                    desired_yaw = _wrap_angle(
                        target_yaw + float(np.clip(_wrap_angle(vec_yaw - target_yaw), -0.42, 0.42))
                    )
                herr = _wrap_angle(desired_yaw - yaw)
                if dist < 0.24 and abs(herr) > 2.15:
                    tgt_vx = -0.20
                    tgt_wz = float(np.clip(0.70 * _wrap_angle(target_yaw - yaw), -0.34, 0.34))
                elif abs(herr) > 0.52 and dist >= 0.16:
                    tgt_vx = 0.0
                    tgt_wz = float(np.clip(0.95 * herr, -0.44, 0.44))
                else:
                    tgt_vx = float(np.clip(0.80 * dist, 0.18, 0.38))
                    if bumper < 0.44:
                        tgt_vx = min(tgt_vx, 0.20)
                    tgt_wz = float(np.clip(0.90 * herr, -0.40, 0.40))
            else:
                herr = _wrap_angle(target_yaw - yaw)
                tgt_vx = 0.0
                tgt_wz = float(np.clip(0.80 * herr, -0.34, 0.34))

            # Safety bumper: brake smoothly well before shins/toes hit countertops/appliances
            if tgt_vx > 0.05 and bumper < 0.30:
                tgt_vx = 0.0
                if dist < 0.34 and yaw_err < 0.35:
                    break

            ok, prev_vx, prev_wz = self._step_slew(prev_vx, prev_wz, tgt_vx, tgt_wz)
            if not ok:
                return False

        if settle:
            # Ensure heading alignment before handing over to arm manipulation
            for _ in range(45):
                with self._sim_lock:
                    unstable = self.sys0.fell() or self.sys0.recovering
                    yaw = heading_z(self.data.qpos[3:7])
                if self._aborted() or unstable:
                    return False
                herr = _wrap_angle(target_yaw - yaw)
                if abs(herr) < 0.10 and abs(prev_vx) < 0.06:
                    break
                tgt_wz = 0.0 if abs(herr) < 0.10 else float(np.clip(0.85 * herr, -0.36, 0.36))
                ok, prev_vx, prev_wz = self._step_slew(prev_vx, prev_wz, 0.0, tgt_wz)
                if not ok:
                    return False

            if not self._ramp_to_stop(prev_vx, prev_wz, wait_hold=True, max_hold=30):
                return False

        with self._sim_lock:
            yaw_final = heading_z(self.data.qpos[3:7])
            fell_final = self.sys0.fell()
            p_final = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
        yaw_ok = (not settle) or abs(_wrap_angle(target_yaw - yaw_final)) < 0.42
        dist_lim = 0.32 if settle else 0.22
        return (
            (not fell_final)
            and yaw_ok
            and float(np.linalg.norm(tgt - p_final)) < dist_lim
        )

    def walk_to(self, target_xy: np.ndarray, target_yaw: float, max_steps: int = 175) -> bool:
        """Closed-loop bumper-guarded locomotion using G1LegPolicy, waiting for full gait->stand fade."""
        tgt = np.asarray(target_xy, dtype=np.float64)
        with self._sim_lock:
            p0 = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
        # If already near the counter wall and shifting laterally, retreat to open floor first (L-path)
        dy = float(tgt[1] - p0[1])
        if p0[0] > 0.16 and abs(dy) > 0.24:
            vx, wz = 0.0, 0.0
            for _ in range(28):
                with self._sim_lock:
                    px = float(self.data.xpos[self.sys0.pelvis_id, 0])
                if px < -0.15:
                    break
                ok, vx, wz = self._step_slew(vx, wz, -0.22, 0.0, max_dvx=0.07)
                if not ok:
                    return False
            if not self._ramp_to_stop(vx, wz, wait_hold=False):
                return False
            # Aim beyond tgt[1] along travel dir to absorb ~90deg yaw-turn lateral drift in open floor
            corridor_y = float(tgt[1]) + (0.50 if dy > 0.0 else -0.50)
            corridor = np.array([min(-0.14, float(tgt[0]) - 0.45), corridor_y], dtype=np.float64)
            if not self._walk_direct(corridor, float(target_yaw), max_steps=165, stop_dist=0.14, settle=False):
                return False
            # Align heading toward target_yaw smoothly in open floor before approaching the counter
            wz_a = 0.0
            for _ in range(65):
                with self._sim_lock:
                    unstable = self.sys0.fell() or self.sys0.recovering
                    yaw = heading_z(self.data.qpos[3:7])
                if self._aborted() or unstable:
                    return False
                herr = _wrap_angle(target_yaw - yaw)
                if abs(herr) < 0.10:
                    break
                tgt_wz = float(np.clip(0.85 * herr, -0.36, 0.36))
                ok, _, wz_a = self._step_slew(0.0, wz_a, 0.0, tgt_wz, max_dwz=0.10)
                if not ok:
                    return False
            if not self._ramp_to_stop(0.0, wz_a, wait_hold=False):
                return False
        return self._walk_direct(tgt, target_yaw, max_steps=max_steps, stop_dist=0.08, settle=True)

    def _choose_stance(self, pt: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Select collision-free, CoM-balanced standing pose (xy, yaw, squat) checked in MuJoCo scratch."""
        with self._sim_lock:
            pelvis0 = np.asarray(self.data.xpos[self.sys0.pelvis_id, :2], dtype=np.float64).copy()
            self.stance_probe.qpos[:] = self.data.qpos
            self.stance_probe.qvel[:] = self.data.qvel
        to_p = pt[:2] - pelvis0
        yaw_face = math.atan2(to_p[1], to_p[0]) if float(np.linalg.norm(to_p)) > 0.15 else 0.0
        if abs(_wrap_angle(yaw_face)) < 1.48 and pt[0] > 0.55:
            yaw_face = 0.0

        fwd = np.array([math.cos(yaw_face), math.sin(yaw_face)], dtype=np.float64)
        left = np.array([-fwd[1], fwd[0]], dtype=np.float64)

        best = (1e9, 1e9, pelvis0, yaw_face, 0.05)
        for d_front in (0.38, 0.42, 0.46, 0.34):
            for d_left in (0.14, 0.22, 0.06):
                for sq in (0.05, 0.25, 0.45):
                    xy = pt[:2] - d_front * fwd + d_left * left
                    self.stance_probe.qpos[0:2] = xy
                    self.stance_probe.qpos[3:7] = [math.cos(0.5 * yaw_face), 0.0, 0.0, math.sin(0.5 * yaw_face)]
                    mujoco.mj_kinematics(self.model, self.stance_probe)
                    bumper = self._front_bumper_clearance(self.stance_probe)
                    floor_clr = self._stance_floor_clearance(self.stance_probe)
                    q_sol, err = solve_whole_body_ik(
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
                    self.scratch.qpos[:] = self.stance_probe.qpos
                    self.scratch.qpos[self.sys0.qadr[:N_ACT]] = q_sol
                    mujoco.mj_forward(self.model, self.scratch)
                    com_margin = float(np.linalg.norm(com_err_xy(self.scratch, self.sys0.pelvis_id, self.sys0.r_fg, self.sys0.l_fg)))
                    bumper_pen = max(0.0, 0.26 - bumper) * 0.9
                    floor_pen = max(0.0, 0.28 - floor_clr) * 1.2
                    score = float(err) + 0.40 * com_margin + bumper_pen + floor_pen
                    if score < best[0]:
                        best = (score, float(err), xy, yaw_face, sq)
        return best[2], best[3], best[4], best[1]

    def _rehearse_primitive(
        self, p: np.ndarray, d_world: np.ndarray, R: np.ndarray, squat: float, prim: str
    ) -> Tuple[bool, float, float]:
        """Mentally test contact reachability & upright CoM balance of the primitive in self.scratch."""
        with self._sim_lock:
            if self.sys0.fell():
                return False, 1.0, 1.0
            self.stance_probe.qpos[:] = self.data.qpos
            self.stance_probe.qvel[:] = self.data.qvel
            yaw = heading_z(self.data.qpos[3:7])
        fwd_w = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        travel = 0.20 if prim == "pull_back" else 0.14
        waypoints = [
            p - 0.09 * fwd_w + np.array([0.0, 0.0, 0.04]),
            p,
            p + d_world * travel,
        ]
        contact_ik = 0.0
        max_ik = 0.0
        max_com = 0.0
        q_cur = None
        for idx, wp in enumerate(waypoints):
            q_sol, err = solve_whole_body_ik(
                self.model,
                self.stance_probe,
                self.scratch,
                self.sys0.qadr,
                self.sys0.vadr,
                self.sys0.wrist_r_id,
                wp,
                R,
                squat=squat,
                q_init=q_cur,
            )
            q_cur = q_sol
            self.scratch.qpos[:] = self.stance_probe.qpos
            self.scratch.qpos[self.sys0.qadr[:N_ACT]] = q_sol
            mujoco.mj_forward(self.model, self.scratch)
            com_err = float(np.linalg.norm(com_err_xy(self.scratch, self.sys0.pelvis_id, self.sys0.r_fg, self.sys0.l_fg)))
            if idx <= 1:
                contact_ik = max(contact_ik, float(err))
            max_ik = max(max_ik, float(err))
            max_com = max(max_com, com_err)
        ok = contact_ik < 0.14 and max_ik < 0.28 and max_com < 0.14
        return ok, max_ik, max_com

    def _move_hook(
        self,
        tgt_pos: np.ndarray,
        tgt_mat: np.ndarray,
        squat: float,
        grip: float,
        raw_cur: np.ndarray,
        steps: int,
        traj: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ) -> Tuple[np.ndarray, float]:
        cur = raw_cur.copy()
        for s in range(max(1, steps)):
            frac = (s + 1) / max(1, steps)
            with self._sim_lock:
                self.stance_probe.qpos[:] = self.data.qpos
                hook_now = self.sys0.hook_pos("right").copy()
                com_b = com_err_xy(self.data, self.sys0.pelvis_id, self.sys0.r_fg, self.sys0.l_fg).copy()
            # Stop driving the arm deeper into rigid furniture if reaction forces shove CoM off-balance
            if com_b[0] < -0.022 or com_b[0] > 0.042:
                break
            err_vec = tgt_pos - hook_now
            eff_pos = tgt_pos + (0.45 * np.clip(err_vec, -0.06, 0.06) if frac > 0.30 else 0.0)
            q_init = np.zeros(N_ACT, dtype=np.float32)
            q_init[12] = float(cur[3])
            q_init[14] = float(cur[4])
            q_init[22:29] = cur[5:12]
            q_init[15:22] = cur[12:19]
            q_sol, _ = solve_whole_body_ik(
                self.model,
                self.stance_probe,
                self.scratch,
                self.sys0.qadr,
                self.sys0.vadr,
                self.sys0.wrist_r_id,
                eff_pos,
                tgt_mat,
                squat=squat,
                q_init=q_init,
            )
            nxt = default_raw_action()
            nxt[0:2] = 0.0
            nxt[2] = float(squat)
            nxt[3] = float(q_sol[12])
            nxt[4] = float(np.clip(q_sol[14], -0.10, 0.18))
            nxt[5:12] = q_sol[22:29]
            nxt[19] = float(grip)
            alpha = min(1.0, 2.0 / max(steps - s, 1))
            cur = (cur + (nxt - cur) * alpha).astype(np.float32)
            cur[0:2] = 0.0
            cur[19] = float(grip)
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

        # Mental simulation in scratch BEFORE moving physical robot
        safe_ok, rehearse_ik, rehearse_com = self._rehearse_primitive(p, d_world, R, squat, prim)
        if not safe_ok:
            self._log(
                "warn",
                f"[Репетиция scratch] Отклоняю `{prim}`: IK={rehearse_ik*100:.1f}см, CoM={rehearse_com*100:.1f}см",
            )
            return False, f"небезопасно в симуляции (IK={rehearse_ik*100:.0f}см, CoM={rehearse_com*100:.0f}см)", []

        before_rgb = self._render_head().copy()
        traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        raw = default_raw_action()
        raw[0:2] = 0.0
        raw[2] = float(squat)

        fwd_w = Rz @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        p_eff = p
        if prim == "pull_back":
            pre = p - 0.09 * fwd_w + np.array([0.0, 0.0, 0.05])
            raw, _ = self._move_hook(pre, R, squat, 0.0, raw, steps=8, traj=traj)
            over = p + 0.022 * fwd_w + np.array([0.0, 0.0, 0.044])
            raw, _ = self._move_hook(over, R, squat, 0.0, raw, steps=8, traj=traj)
            contact = p + 0.022 * fwd_w - np.array([0.0, 0.0, 0.020])
            raw, _ = self._move_hook(contact, R, squat, 0.8, raw, steps=6, traj=traj)
            p_eff = contact
        elif prim == "press_down":
            pre = p - 0.045 * fwd_w + np.array([0.0, 0.0, 0.075])
            raw, _ = self._move_hook(pre, R, squat, 0.35, raw, steps=10, traj=traj)
            over = p + np.array([0.0, 0.0, 0.025])
            raw, _ = self._move_hook(over, R, squat, 0.35, raw, steps=10, traj=traj)
            p_eff = p + np.array([0.0, 0.0, 0.014])
        else:
            pre = p - 0.08 * d_world + np.array([0.0, 0.0, 0.02])
            raw, _ = self._move_hook(pre, R, squat, 0.0, raw, steps=8, traj=traj)
            raw, _ = self._move_hook(p, R, squat, 0.0, raw, steps=6, traj=traj)

        grip = PRIM_GRIP.get(prim, 0.0)
        raw[0:2] = 0.0
        raw[19] = grip
        for _ in range(6 if grip > 0.0 else 2):
            if not self._run_step(raw, traj, tuck_left=True, tuck_right=False):
                return False, "потеряно равновесие или остановлено", traj

        travel = 0.16 if prim == "pull_back" else (0.22 if prim == "push_forward" else (0.105 if prim == "press_down" else 0.18))
        n_seg = 12 if prim == "pull_back" else (5 if prim == "press_down" else 6)
        seg_steps = 1 if prim == "pull_back" else (3 if prim == "press_down" else 2)
        for k in range(1, n_seg + 1):
            tgt = p_eff + d_world * (travel * k / n_seg)
            raw, err = self._move_hook(tgt, R, squat, grip, raw, steps=seg_steps, traj=traj)
            with self._sim_lock:
                fell = self.sys0.fell()
                wrench = float(np.linalg.norm(self.data.cfrc_ext[self.sys0.wrist_r_id]))
                com_err = float(
                    np.linalg.norm(com_err_xy(self.data, self.sys0.pelvis_id, self.sys0.r_fg, self.sys0.l_fg))
                )
                vb_x = float(cy * self.data.qvel[0] + sy * self.data.qvel[1])
            if self._aborted() or fell:
                break
            # Closed-loop tactile & balance compliance: yield before rigid reaction force or CoM drift tips G1
            if prim in ("press_down", "push_forward") and (err > 0.11 or wrench > 18.0 or com_err > 0.10):
                break
            if prim == "pull_back" and (err > 0.14 or com_err > 0.048 or vb_x < -0.18):
                break
            if err > 0.24 or com_err > 0.13:
                break

        mid_rgb = self._render_head().copy()

        raw[19] = 0.0
        with self._sim_lock:
            cur_hook = self.sys0.hook_pos("right").copy()
            fell_mid = self.sys0.fell()
        if not fell_mid:
            stand = default_raw_action()
            if prim == "pull_back":
                # Blend arm/torso to stand while absorbing backward door-latch recoil, then settle briefly
                vx_b = 0.0
                for k in range(1, 9):
                    frac = k / 8.0
                    blend = (raw + (stand - raw) * frac).astype(np.float32)
                    with self._sim_lock:
                        vb_x = float(cy * self.data.qvel[0] + sy * self.data.qvel[1])
                    tgt_vxb = float(np.clip(vb_x * 0.45, -0.18, 0.0)) if vb_x < -0.14 else 0.0
                    vx_b = float(np.clip(tgt_vxb, vx_b - 0.07, vx_b + 0.07))
                    blend[0] = vx_b
                    blend[1] = 0.0
                    if not self._run_step(blend, None, tuck_left=True, tuck_right=(k >= 4)):
                        break
                for _ in range(4):
                    if abs(vx_b) < 0.04:
                        break
                    ok, vx_b, _ = self._step_slew(vx_b, 0.0, 0.0, 0.0)
                    if not ok:
                        break
                self._wait_gait_hold(None, max_ticks=14, speed_tol=0.12, min_ticks=5)
                self._safe_back_up(target_clear=0.45, max_ticks=10)
            else:
                retract = cur_hook - 0.09 * fwd_w + np.array([0.0, 0.0, 0.05])
                raw, _ = self._move_hook(retract, R, 0.05, 0.0, raw, steps=5, traj=None)
                for k in range(1, 8):
                    frac = k / 7.0
                    blend = (raw + (stand - raw) * frac).astype(np.float32)
                    blend[0:2] = 0.0
                    self._run_step(blend, None, tuck_left=True, tuck_right=(k >= 5))

        after_rgb = self._render_head().copy()
        with self._sim_lock:
            fell_end = fell_mid or self.sys0.fell() or self.sys0.recovering
        if fell_end:
            return False, "потеряно равновесие", traj
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
                if vx > 0.05 and self._front_bumper_clearance() < 0.34:
                    break
                if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            return self._wait_gait_hold(traj, max_ticks=14), traj

        if skill == "turn":
            wz = -0.55 if "right" in direction else 0.55
            angle_deg = float(params.get("angle") or 60)
            dur_ticks = int(np.clip(abs(angle_deg) / 6.0, 8, 30))
            raw[1] = wz
            for _ in range(dur_ticks):
                if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                    return False, traj
            return self._wait_gait_hold(traj, max_ticks=14), traj

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

        for _ in range(10):
            if not self._run_step(raw, traj, tuck_left=True, tuck_right=True):
                return False, traj
        return True, traj

    def _already_done_result(self, verdict: Dict[str, Any], command: str) -> Dict[str, Any]:
        self.marker_pos = None
        why = verdict.get("status") or "состояние сцены уже соответствует команде"
        self._log("info", f"[VLM Вердикт] Цель «{command}» уже выполнена ({why}).")
        return {"ok": True, "already_done": True, "mode": "already_done", "samples": []}

    def _abort_or_fall_result(self, rgb0: np.ndarray, tried_motion: str, what: str) -> Dict[str, Any]:
        """Distinguish operator stop from locomotion fall so failures trigger autonomous recovery + retry."""
        with self._sim_lock:
            fell_now = self.sys0.fell() or self.sys0.recovering
        if self._aborted() and not fell_now:
            return {"ok": False, "reason": "Остановлено", "samples": []}
        return {
            "ok": False,
            "reason": "walk_aborted",
            "samples": [],
            "failure_info": {
                "before_rgb": rgb0,
                "after_rgb": self._render_head(),
                "tried_motions": [tried_motion],
                "ik_err_cm": 0.0,
                "fell": True,
                "what": what,
            },
        }

    def execute(self, command: str, *, lessons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Full System 2 execution pipeline returning success status, Wake chunks, and failure diagnostics."""
        self.marker_pos = None
        with self._sim_lock:
            fell0 = self.sys0.fell() or self.sys0.recovering
        if fell0:
            return {"ok": False, "reason": "робот лежит", "samples": []}

        # Back up gently into open floor and wait for full gait->stand hold before blocking VLM call
        if self._front_bumper_clearance() < 0.34:
            if not self._safe_back_up(target_clear=0.40, max_ticks=14):
                rgb_now = self._render_head()
                return self._abort_or_fall_result(
                    rgb_now, "back_up", "потеря равновесия при отходе от препятствия перед планированием"
                )

        rgb0 = self._render_head()
        plan = self.planner.plan_command(rgb0, command, lessons=lessons)
        if not plan.get("ok"):
            return {"ok": False, "reason": "VLM недоступна", "samples": []}
        with self._sim_lock:
            fell_plan = self.sys0.fell() or self.sys0.recovering
        if self._aborted() or fell_plan:
            return self._abort_or_fall_result(rgb0, "plan", "потеря равновесия во время распознавания сцены")

        if plan["mode"] == "body_skill":
            self._log("sys2", f"[System 2] Базовый навык: {plan.get('skill')} {plan.get('params')}")
            ok, traj = self._run_body_skill(plan, command)
            samples = self._build_chunks(traj, command)
            if ok:
                return {"ok": True, "mode": "body_skill", "reason": "", "samples": samples}
            with self._sim_lock:
                fell_now = self.sys0.fell()
            return {
                "ok": False,
                "mode": "body_skill",
                "reason": "потеряно равновесие или остановлено",
                "samples": samples,
                "failure_info": {
                    "before_rgb": rgb0,
                    "after_rgb": self._render_head(),
                    "tried_motions": [str(plan.get("skill") or "body_skill")],
                    "ik_err_cm": 0.0,
                    "fell": fell_now,
                    "what": "потеря равновесия при выполнении движения корпуса",
                },
            }

        if plan.get("already_done"):
            return self._already_done_result(plan, command)

        # Wide ~360° visual search if target is not currently visible (or initial pixel missed objects)
        found = plan if (plan.get("visible") and plan.get("uv") and self.uv_to_world(*plan["uv"]) is not None) else None
        if found is None:
            wz_tgt = 0.34 if plan.get("search_dir") == "left" else -0.34
            self._log("sys2", f"[System 2] Цель не в кадре, осматриваюсь ({plan.get('search_dir')})...")
            if self._front_bumper_clearance() < 0.42:
                if not self._safe_back_up(target_clear=0.45, max_ticks=14):
                    return self._abort_or_fall_result(rgb0, "search", "потеря равновесия при отходе для обзора")
            for _ in range(10):
                wz_cur = 0.0
                for _ in range(12):
                    ok, _, wz_cur = self._step_slew(0.0, wz_cur, 0.0, wz_tgt, max_dwz=0.10)
                    if not ok:
                        return self._abort_or_fall_result(rgb0, "search", "потеря равновесия при поиске цели поворотом")
                # Smoothly ramp to full stand hold BEFORE blocking on VLM HTTP call
                if not self._ramp_to_stop(0.0, wz_cur, wait_hold=True, max_hold=20):
                    return self._abort_or_fall_result(rgb0, "search", "потеря равновесия после поворота при поиске цели")
                g = self.planner.ground(self._render_head(), command, lessons=lessons)
                if g.get("already_done"):
                    self._wait_gait_hold(None, max_ticks=12)
                    return self._already_done_result(g, command)
                if g.get("visible") and g.get("uv") and self.uv_to_world(*g["uv"]) is not None:
                    found = g
                    break

        if found is None or not found.get("uv"):
            self._wait_gait_hold(None, max_ticks=12)
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
            self.marker_pos = None
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
                    "what": "не удалось безопасно подойти к выбранной точке",
                },
            }

        # Re-ground closer up once firmly planted
        g2 = self.planner.ground(self._render_head(), command, lessons=lessons)
        if g2.get("already_done"):
            return self._already_done_result(g2, command)
        motions = found["motions"]
        if g2.get("visible") and g2.get("uv"):
            hit2 = self.uv_to_world(*g2["uv"])
            if hit2 is not None and float(np.linalg.norm(hit2[0][:2] - pt[:2])) < 0.75:
                pt2 = hit2[0]
                horiz_shift = float(np.linalg.norm(pt2[:2] - pt[:2]))
                pt = pt2
                self.marker_pos = pt.copy()
                stance2_xy, stance2_yaw, squat, ik_err = self._choose_stance(pt)
                motions = g2["motions"]
                self._log("sys2", f"[Grounding вблизи] {np.round(pt, 2)} · движения: {motions[:3]}")
                if horiz_shift > 0.26:
                    self.walk_to(stance2_xy, stance2_yaw, max_steps=80)

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
