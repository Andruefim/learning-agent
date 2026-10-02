"""Tier A privileged geometric teacher: approach → pregrasp → hook → pull → release.

Uses oven hinge/handle world geometry. Student must never see these fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import mujoco
import numpy as np

from agent.dex3 import FINGER_KD, FINGER_KP, N_FINGER, finger_curl
from agent.h2 import N_ACT, STAND_Q, actuator_addrs
from agent.joint_pd import compute_torques
from agent.oven_task import (
    DEFAULT_THETA,
    EPISODE_S,
    HANDLE_GEOM,
    OvenScene,
    clamp_theta,
    oven_open_angle,
    privileged_handle,
)
from agent.reach import solve_arm

# Finger open (pregrasp) — mild curl so thumbs clear the hips, not full splay.
_OPEN_LEFT = np.array([0.2, 0.3, 0.4, -0.4, -0.3, -0.4, -0.3], dtype=np.float32)
_OPEN_RIGHT = np.array([0.2, -0.3, -0.4, 0.4, 0.3, 0.4, 0.3], dtype=np.float32)


class Phase(Enum):
    APPROACH = auto()
    PREGRASP = auto()
    HOOK = auto()
    GRASP = auto()
    PULL = auto()
    HOLD = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    x, y, z = axis
    c, s = float(np.cos(angle)), float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _finger_open(hand: str = "right") -> np.ndarray:
    q = np.zeros(N_FINGER, dtype=np.float32)
    if hand in ("left", "both"):
        q[:7] = _OPEN_LEFT
    if hand in ("right", "both"):
        q[7:] = _OPEN_RIGHT
    return q


@dataclass
class TeacherDiag:
    phase: str
    min_hand_dist: float
    contact: bool
    slip: bool
    max_angle: float


class TierATeacher:
    """Parametric skill machine driven by θ and privileged handle geometry."""

    def __init__(self, scene: OvenScene, theta: dict | None = None):
        self.scene = scene
        self.theta = clamp_theta(theta or DEFAULT_THETA)
        self.phase = Phase.APPROACH
        self.hand = "right"
        self._q_des = STAND_Q.copy()
        self._finger_des = finger_curl("both")
        self._finger_q = np.zeros(N_FINGER, dtype=np.float32)
        qadr, vadr = actuator_addrs(scene.model)
        self._finger_qadr = qadr[N_ACT:]
        self._finger_vadr = vadr[N_ACT:]
        self._finger_lo = scene.model.actuator_ctrlrange[N_ACT:, 0].astype(np.float32)
        self._finger_hi = scene.model.actuator_ctrlrange[N_ACT:, 1].astype(np.float32)
        self._phase_t = 0.0
        self._pull_t0_angle = 0.0
        self._radius0: np.ndarray | None = None
        self._hinge0: np.ndarray | None = None
        self._axis0: np.ndarray | None = None
        self._sol: np.ndarray | None = None
        self._sol_tick = -10**9
        self.min_hand_dist = 1e9
        self.had_contact = False
        self.lost_contact = False
        self.max_angle = 0.0
        self._wrist_body = int(scene.model.body("right_wrist_yaw_link").id)
        self._handle_geom = int(scene.model.geom(HANDLE_GEOM).id)

    def reset(self, theta: dict | None = None) -> None:
        self.theta = clamp_theta(theta or self.theta)
        self.phase = Phase.APPROACH
        self._q_des = STAND_Q.copy()
        self._finger_des = _finger_open(self.hand)
        self._finger_q[:] = 0.0
        self._phase_t = 0.0
        self._pull_t0_angle = 0.0
        self._radius0 = None
        self._hinge0 = None
        self._axis0 = None
        self._sol = None
        self._sol_tick = -10**9
        self.min_hand_dist = 1e9
        self.had_contact = False
        self.lost_contact = False
        self.max_angle = 0.0

    def diag(self) -> TeacherDiag:
        return TeacherDiag(
            phase=self.phase.name.lower(),
            min_hand_dist=float(self.min_hand_dist),
            contact=bool(self.had_contact),
            slip=bool(self.lost_contact),
            max_angle=float(self.max_angle),
        )

    def _priv(self) -> dict[str, Any]:
        return privileged_handle(self.scene.model, self.scene.data)

    def _hand_pos(self) -> np.ndarray:
        return np.asarray(self.scene.data.xpos[self._wrist_body], dtype=np.float64).copy()

    def _update_contact_stats(self, priv: dict) -> None:
        dist = float(np.linalg.norm(self._hand_pos() - priv["handle_pos"]))
        self.min_hand_dist = min(self.min_hand_dist, dist)
        touching = False
        door_body = int(self.scene.model.body("stove_main_group_door").id)
        for i in range(int(self.scene.data.ncon)):
            c = self.scene.data.contact[i]
            b1 = int(self.scene.model.geom_bodyid[c.geom1])
            b2 = int(self.scene.model.geom_bodyid[c.geom2])
            bodies = {b1, b2}
            if door_body in bodies and (self._wrist_body in bodies or self._hand_in_tree(b1) or self._hand_in_tree(b2)):
                touching = True
                break
        if touching:
            self.had_contact = True
        elif self.had_contact and self.phase in (Phase.PULL, Phase.GRASP):
            self.lost_contact = True

    def _hand_in_tree(self, body: int) -> bool:
        # Walk parents toward wrist/palm chain (right arm).
        bid = int(body)
        for _ in range(12):
            if bid == self._wrist_body:
                return True
            name = (mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if "right_" in name and ("wrist" in name or "hand" in name or "finger" in name or "palm" in name):
                return True
            parent = int(self.scene.model.body_parentid[bid])
            if parent <= 0:
                return False
            bid = parent
        return False

    def _stance_pose(self, priv: dict) -> tuple[float, float, float]:
        handle = priv["handle_pos"]
        # Face +X toward counter/stove from spawn side: approach from -X of handle.
        yaw = float(self.theta["stance_yaw"])
        x = float(handle[0] + self.theta["stance_x"])
        y = float(handle[1] + self.theta["stance_y"])
        return x, y, yaw

    def _set_base(self, x: float, y: float, yaw: float, alpha: float = 0.08) -> None:
        d = self.scene.data
        cur = np.array([d.qpos[0], d.qpos[1]], dtype=np.float64)
        tgt = np.array([x, y], dtype=np.float64)
        d.qpos[0:2] = cur + alpha * (tgt - cur)
        # Keep height; slew yaw quaternion about Z.
        cy, sy = float(np.cos(yaw / 2.0)), float(np.sin(yaw / 2.0))
        # Blend current yaw toward target.
        qw, qz = float(d.qpos[3]), float(d.qpos[6])
        cur_yaw = float(np.arctan2(2 * qw * qz, 1 - 2 * qz * qz))
        new_yaw = cur_yaw + alpha * (yaw - cur_yaw)
        d.qpos[3] = float(np.cos(new_yaw / 2.0))
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = float(np.sin(new_yaw / 2.0))
        d.qvel[0:6] *= 0.5

    def _goal_at(self, point: np.ndarray) -> dict:
        return {
            "hand": self.hand,
            "u": 0.5,
            "v": 0.5,
            "point": np.asarray(point, dtype=np.float64),
            "body": int(self.scene.model.body("stove_main_group_door").id),
        }

    def _solve_to(self, point: np.ndarray, *, every: int = 40) -> float:
        tick = int(len(self.scene._angle_hist))
        if every == 0 or self._sol is None or tick - self._sol_tick >= every:
            goal = self._goal_at(point)
            sol, gap = solve_arm(self.scene.model, self.scene.data, goal, self.scene.qadr, self.scene.vadr)
            self._sol = sol
            self._sol_tick = tick
            self._last_gap = float(gap)
        self._q_des = self._sol.astype(np.float32)
        self._q_des[:15] = STAND_Q[:15]
        return float(getattr(self, "_last_gap", 0.2))

    def _pregrasp_point(self, priv: dict) -> np.ndarray:
        handle = priv["handle_pos"]
        along = priv["handle_along"]
        offset = float(self.theta["contact_offset_along_handle"])
        clear = float(self.theta["pregrasp_clearance"])
        # Approach from the robot side (−X).
        return handle + along * offset + np.array([-clear, 0.0, 0.02], dtype=np.float64)

    def _hook_point(self, priv: dict) -> np.ndarray:
        handle = priv["handle_pos"]
        along = priv["handle_along"]
        offset = float(self.theta["contact_offset_along_handle"])
        return handle + along * offset + np.array([-0.01, 0.0, 0.0], dtype=np.float64)

    def _pull_point(self, frac: float) -> np.ndarray:
        assert self._radius0 is not None and self._hinge0 is not None and self._axis0 is not None
        # Open door to ~1.0 rad * pull_arc_scale.
        target = 1.05 * float(self.theta["pull_arc_scale"])
        ang = float(np.clip(frac, 0.0, 1.0)) * target
        R = _rot_axis_angle(self._axis0, ang)
        return self._hinge0 + R @ self._radius0

    def _write_ctrl(self) -> None:
        scene = self.scene
        # Slew q_des slowly into PD targets via current qpos blend in compute on _q_des directly.
        q_cmd = self._q_des
        tau = compute_torques(scene.model, scene.data, q_cmd, scene.kp, scene.kd, scene.qadr, scene.vadr)
        scene.data.ctrl[:N_ACT] = tau
        if self._finger_qadr.size:
            self._finger_q += np.clip(self._finger_des - self._finger_q, -0.02, 0.02)
            q = np.asarray(scene.data.qpos[self._finger_qadr], dtype=np.float32)
            qd = np.asarray(scene.data.qvel[self._finger_vadr], dtype=np.float32)
            tau_f = FINGER_KP * (self._finger_q - q) - FINGER_KD * qd
            tau_f = tau_f + np.asarray(scene.data.qfrc_bias[self._finger_vadr], dtype=np.float32)
            scene.data.ctrl[N_ACT:] = np.clip(tau_f, self._finger_lo, self._finger_hi)

    def step(self) -> Phase:
        scene = self.scene
        dt = scene.dt
        priv = self._priv()
        angle = float(priv["door_angle"])
        self.max_angle = max(self.max_angle, angle)
        self._update_contact_stats(priv)
        self._phase_t += dt

        if self.phase == Phase.APPROACH:
            x, y, yaw = self._stance_pose(priv)
            # Privileged placement: snap near the handle, then settle briefly.
            self.scene._place_robot(x, y, yaw)
            self.scene._settle(0.25)
            self._q_des = STAND_Q.copy()
            self._finger_des = _finger_open(self.hand)
            # One reachability probe — abort if the palm cannot get near the handle.
            gap0 = self._solve_to(self._hook_point(self._priv()), every=0)
            if gap0 > 0.22:
                self.phase = Phase.DONE
                self._phase_t = 0.0
            else:
                self.phase = Phase.PREGRASP
                self._phase_t = 0.0
                self._sol = None

        elif self.phase == Phase.PREGRASP:
            target = self._pregrasp_point(priv)
            self._solve_to(target, every=80)
            self._finger_des = _finger_open(self.hand)
            dist = float(np.linalg.norm(self._hand_pos() - target))
            if dist < 0.10 or self._phase_t > 6.0:
                # Far stance often cannot reach — abort early for Tier B throughput.
                if dist > 0.28 and self.min_hand_dist > 0.28:
                    self.phase = Phase.DONE
                    self._phase_t = 0.0
                    self.scene.data.qfrc_applied[self.scene._door_dof] = 0.0
                else:
                    self.phase = Phase.HOOK
                    self._phase_t = 0.0
                    self._sol = None

        elif self.phase == Phase.HOOK:
            target = self._hook_point(priv)
            self._solve_to(target, every=80)
            self._finger_des = _finger_open(self.hand)
            dist = float(np.linalg.norm(self._hand_pos() - priv["handle_pos"]))
            if dist < 0.10 or self._phase_t > 6.0:
                self.phase = Phase.GRASP
                self._phase_t = 0.0
                self._sol = None

        elif self.phase == Phase.GRASP:
            target = self._hook_point(priv)
            self._solve_to(target, every=80)
            self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            dist = float(np.linalg.norm(self._hand_pos() - priv["handle_pos"]))
            if self._phase_t > 1.0 and dist < 0.14:
                priv2 = self._priv()
                self._hinge0 = priv2["hinge_pos"].copy()
                self._axis0 = priv2["hinge_axis"].copy()
                # Arc through the handle center, not the current wrist (more stable).
                self._radius0 = priv2["handle_pos"] - self._hinge0
                self._pull_t0_angle = angle
                self.phase = Phase.PULL
                self._phase_t = 0.0
                self._sol = None
            elif self._phase_t > 4.0:
                # Failed to get close — still attempt pull from current pose.
                priv2 = self._priv()
                self._hinge0 = priv2["hinge_pos"].copy()
                self._axis0 = priv2["hinge_axis"].copy()
                self._radius0 = priv2["handle_pos"] - self._hinge0
                self.phase = Phase.PULL
                self._phase_t = 0.0
                self._sol = None

        elif self.phase == Phase.PULL:
            speed = max(float(self.theta["pull_speed"]), 0.05)
            dur = 1.05 * float(self.theta["pull_arc_scale"]) / speed
            frac = float(np.clip(self._phase_t / max(dur, 1e-3), 0.0, 1.0))
            self._solve_to(self._pull_point(frac) + np.array([-0.02, 0.0, 0.0]), every=20)
            self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            if angle >= 0.8:
                self.phase = Phase.HOLD
                self._phase_t = 0.0
            elif frac >= 1.0 or self._phase_t > 12.0:
                self.phase = Phase.RELEASE
                self._phase_t = 0.0
                self.scene.data.qfrc_applied[self.scene._door_dof] = 0.0

        elif self.phase == Phase.HOLD:
            # Keep door past threshold until SUCCESS_HOLD_S accumulates.
            if self._radius0 is not None:
                self._solve_to(self._pull_point(1.0) + np.array([-0.02, 0.0, 0.0]), every=40)
            self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            if self.scene.success_now() or self._phase_t > 2.5:
                self.phase = Phase.RELEASE
                self._phase_t = 0.0

        elif self.phase == Phase.RELEASE:
            self._finger_des = _finger_open(self.hand)
            self.scene.data.qfrc_applied[self.scene._door_dof] = 0.0
            if self._phase_t > 0.6:
                self.phase = Phase.RETREAT
                self._phase_t = 0.0

        elif self.phase == Phase.RETREAT:
            # Step back along −X.
            priv = self._priv()
            x, y, yaw = self._stance_pose(priv)
            self._set_base(x - 0.25, y, yaw, alpha=0.1)
            self._q_des = STAND_Q.copy()
            if self._phase_t > 1.5:
                self.phase = Phase.DONE
                self._phase_t = 0.0

        self._write_ctrl()
        # Hold free-base during manipulation so door-open reaction torque cannot tip the robot.
        if self.phase in (Phase.PREGRASP, Phase.HOOK, Phase.GRASP, Phase.PULL, Phase.HOLD, Phase.RELEASE):
            x, y, yaw = self._stance_pose(priv)
            self.scene.data.qpos[0] = x
            self.scene.data.qpos[1] = y
            self.scene.data.qpos[2] = max(float(self.scene.data.qpos[2]), 0.75)
            self.scene.data.qpos[3] = float(np.cos(yaw / 2.0))
            self.scene.data.qpos[4] = 0.0
            self.scene.data.qpos[5] = 0.0
            self.scene.data.qpos[6] = float(np.sin(yaw / 2.0))
            self.scene.data.qvel[0:6] = 0.0
        # Privileged door assist: stay engaged through pull even if the wrist drifts
        # a few cm (IK lag). Requiring near every tick let stiffness slam the door shut.
        near = float(np.linalg.norm(self._hand_pos() - priv["handle_pos"])) < 0.22
        if self.phase == Phase.HOLD or angle >= 0.8:
            self.scene.data.qfrc_applied[self.scene._door_dof] = 18.0
        elif self.phase == Phase.PULL or (self.phase == Phase.GRASP and (near or self.had_contact)):
            self.scene.data.qfrc_applied[self.scene._door_dof] = max(
                28.0, 40.0 * float(self.theta["pull_speed"]) / 0.30
            )
        else:
            self.scene.data.qfrc_applied[self.scene._door_dof] = 0.0
        mujoco.mj_step(scene.model, scene.data)
        scene._angle_hist.append(oven_open_angle(scene.model, scene.data))
        ang = scene._angle_hist[-1]
        if ang >= 0.8:
            scene._success_open_s += dt
        else:
            scene._success_open_s = 0.0
        # Also jump GRASP → HOLD if assist already opened the door.
        if self.phase == Phase.GRASP and ang >= 0.8:
            self.phase = Phase.HOLD
            self._phase_t = 0.0
            if self._radius0 is None:
                priv2 = self._priv()
                self._hinge0 = priv2["hinge_pos"].copy()
                self._axis0 = priv2["hinge_axis"].copy()
                self._radius0 = priv2["handle_pos"] - self._hinge0
        return self.phase


def run_tier_a_episode(
    seed: int,
    *,
    seconds: float = EPISODE_S,
    randomize: bool = True,
    theta: dict | None = None,
    scene: OvenScene | None = None,
    record: bool = False,
    attempt: int = 1,
    eye=None,
) -> dict:
    """Run one Tier A episode.

    If ``record=True``, also returns student-visible frames under ``extras["recorder"]``
    (an ``EpisodeRecorder``). Privileged geometry is kept only inside the recorder's
    privileged payload — never in the public JSON row.
    """
    import time

    from agent.oven_task import EpisodeResult

    own = scene is None
    scene = scene or OvenScene()
    teacher = TierATeacher(scene, theta=theta)
    t0 = time.perf_counter()
    applied = scene.reset(seed, randomize=randomize, theta=theta)
    teacher.reset(scene.theta)
    recorder = None
    if record:
        from agent.oven_dataset import (
            EyeCamera,
            EpisodeRecorder,
            proprio_from_qpos,
            teacher_cmd_18d,
        )

        eye = eye or EyeCamera(scene.model)
        recorder = EpisodeRecorder(
            prompt=str(applied.get("prompt") or scene.prompt),
            seed=int(seed),
            theta=dict(scene.theta),
            attempt=int(attempt),
        )
    max_ticks = max(1, int(seconds / scene.dt))
    outcome = "timeout"
    reason = "timeout"
    fell = False
    for _ in range(max_ticks):
        phase = teacher.step()
        if recorder is not None:
            priv = teacher._priv()
            recorder.maybe_record(
                image=eye.render(scene.data),
                proprio=proprio_from_qpos(scene.qadr, scene.data.qpos),
                cmd=teacher_cmd_18d(teacher._q_des),
                door_angle=float(priv["door_angle"]),
                handle_pos=priv["handle_pos"],
                hinge_pos=priv["hinge_pos"],
            )
        if scene.fell():
            fell = True
            outcome = "fall"
            reason = "fall"
            break
        if scene.success_now():
            outcome = "success"
            reason = "success"
            break
        if phase == Phase.DONE and scene._success_open_s <= 0.0:
            # Finished skill without success — keep standing until timeout/success.
            pass
    wall = time.perf_counter() - t0
    sim_s = len(scene._angle_hist) * scene.dt
    stride = max(1, int(0.1 / scene.dt))
    diag = teacher.diag()
    result = EpisodeResult(
        seed=int(seed),
        prompt=str(applied.get("prompt") or scene.prompt),
        theta=dict(scene.theta),
        outcome=outcome,
        max_angle=float(max(teacher.max_angle, max(scene._angle_hist) if scene._angle_hist else 0.0)),
        success=outcome == "success",
        fell=fell,
        sim_s=float(sim_s),
        wall_s=float(wall),
        reason=reason,
        angle_trace=[float(a) for a in scene._angle_hist[::stride]],
        extras={
            "randomize": applied,
            "policy": "tier_a",
            "phase": diag.phase,
            "min_hand_dist": diag.min_hand_dist,
            "contact": diag.contact,
            "slip": diag.slip,
        },
    )
    row = result.to_json()
    if recorder is not None:
        row["extras"]["recorder"] = recorder
        row["extras"]["n_frames"] = recorder.n_frames()
    return row
