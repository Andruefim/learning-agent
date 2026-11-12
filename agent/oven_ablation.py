"""Day A no-assist ablation stages (joint-PD teacher path; L3 not measured)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import mujoco
import numpy as np

from agent.dex3 import FINGER_KD, FINGER_KP, N_FINGER, finger_curl
from agent.h2 import N_ACT, STAND_Q, actuator_addrs
from agent.joint_pd import compute_torques
from agent.oven_rules import (
    ContactOpenTracker,
    Rule7State,
    door_params,
    hand_geom_ids,
    robot_body_ids,
    robot_dof_ids,
)
from agent.oven_task import (
    DEFAULT_THETA,
    DOOR_JOINT,
    HANDLE_GEOM,
    OvenScene,
    clamp_theta,
    oven_open_angle,
    privileged_handle,
)
from agent.oven_teacher import Phase, TierATeacher, _finger_open, _rot_axis_angle
from agent.reach import solve_arm

BaseMode = Literal["frozen", "free"]
Stage = Literal["s1_weld", "s2_hook", "s3_honest"]


@dataclass
class AblationResult:
    seed: int
    stage: str
    base: str
    scene_mod: str
    success: bool
    fell: bool
    invalid_episode: bool
    invalid_reasons: list[str]
    diagnostic_cheat: bool
    max_angle: float
    sim_s: float
    wall_s: float
    contact: dict
    door: dict
    extras: dict

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "stage": self.stage,
            "base": self.base,
            "scene_mod": self.scene_mod,
            "success": self.success,
            "fell": self.fell,
            "invalid_episode": self.invalid_episode,
            "invalid_reasons": self.invalid_reasons,
            "diagnostic_cheat": self.diagnostic_cheat,
            "max_angle": self.max_angle,
            "sim_s": self.sim_s,
            "wall_s": self.wall_s,
            "contact": self.contact,
            "door": self.door,
            "l3_path": "not_measured",
            "balance": self.extras.get("balance", ""),
            "execution_path": "joint_pd_teacher",
            "extras": {k: v for k, v in self.extras.items() if k != "balance"},
        }


def _wilson(n_ok: int, n: int, z: float = 1.96) -> list[float]:
    n = max(int(n), 1)
    phat = n_ok / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return [max(0.0, center - half), min(1.0, center + half)]


class HonestTeacher(TierATeacher):
    """Tier A skills without door qfrc assist. Optional base freeze."""

    def __init__(self, scene: OvenScene, theta: dict | None = None, *, freeze_base: bool = True, open_fingers: bool = False):
        super().__init__(scene, theta=theta)
        self.freeze_base = bool(freeze_base)
        self.open_fingers = bool(open_fingers)

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
            self.scene._place_robot(x, y, yaw)
            self.scene._settle(0.25)
            self._q_des = STAND_Q.copy()
            self._finger_des = _finger_open(self.hand)
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
                if dist > 0.28 and self.min_hand_dist > 0.28:
                    self.phase = Phase.DONE
                    self._phase_t = 0.0
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
            if self.open_fingers:
                self._finger_des = _finger_open(self.hand)
            else:
                self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            dist = float(np.linalg.norm(self._hand_pos() - priv["handle_pos"]))
            if (self._phase_t > 1.0 and dist < 0.14) or self._phase_t > 4.0:
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
            # Tangential pull along door arc.
            self._solve_to(self._pull_point(frac) + np.array([-0.02, 0.0, 0.0]), every=20)
            if self.open_fingers:
                self._finger_des = _finger_open(self.hand)
            else:
                self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            if angle >= 0.8:
                self.phase = Phase.HOLD
                self._phase_t = 0.0
            elif frac >= 1.0 or self._phase_t > 12.0:
                self.phase = Phase.RELEASE
                self._phase_t = 0.0

        elif self.phase == Phase.HOLD:
            if self._radius0 is not None:
                self._solve_to(self._pull_point(1.0) + np.array([-0.02, 0.0, 0.0]), every=40)
            if self.open_fingers:
                self._finger_des = _finger_open(self.hand)
            else:
                self._finger_des = finger_curl(self.hand) * float(self.theta["grip_close"])
            if self._phase_t > 2.5:
                self.phase = Phase.RELEASE
                self._phase_t = 0.0

        elif self.phase == Phase.RELEASE:
            self._finger_des = _finger_open(self.hand)
            if self._phase_t > 0.6:
                self.phase = Phase.RETREAT
                self._phase_t = 0.0

        elif self.phase == Phase.RETREAT:
            if self.freeze_base:
                x, y, yaw = self._stance_pose(priv)
                self._set_base(x - 0.25, y, yaw, alpha=0.1)
            self._q_des = STAND_Q.copy()
            if self._phase_t > 1.5:
                self.phase = Phase.DONE
                self._phase_t = 0.0

        # Always clear door assist — Rule 7.
        scene.data.qfrc_applied[scene._door_dof] = 0.0
        self._write_ctrl()
        if self.freeze_base and self.phase in (
            Phase.PREGRASP,
            Phase.HOOK,
            Phase.GRASP,
            Phase.PULL,
            Phase.HOLD,
            Phase.RELEASE,
        ):
            x, y, yaw = self._stance_pose(priv)
            scene.data.qpos[0] = x
            scene.data.qpos[1] = y
            scene.data.qpos[2] = max(float(scene.data.qpos[2]), 0.75)
            scene.data.qpos[3] = float(np.cos(yaw / 2.0))
            scene.data.qpos[4] = 0.0
            scene.data.qpos[5] = 0.0
            scene.data.qpos[6] = float(np.sin(yaw / 2.0))
            scene.data.qvel[0:6] = 0.0
        elif not self.freeze_base:
            # Legs hold stand via PD already in _q_des[:15]=STAND; free joint unconstrained.
            self._q_des[:15] = STAND_Q[:15]

        mujoco.mj_step(scene.model, scene.data)
        scene._angle_hist.append(oven_open_angle(scene.model, scene.data))
        return self.phase


def _weld_wrist_to_handle(
    scene: OvenScene,
    wrist_body: int,
    handle_pos: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    frac: float,
    *,
    q_des: np.ndarray | None = None,
) -> np.ndarray:
    """Diagnostic: pull wrist to door-arc handle point via robot xfrc (+ cached IK)."""
    target_ang = 1.05 * frac
    R = _rot_axis_angle(axis, target_ang)
    radius = handle_pos - hinge
    goal = hinge + R @ radius
    if q_des is None:
        from agent.reach import solve_arm

        goal_spec = {
            "hand": "right",
            "u": 0.5,
            "v": 0.5,
            "point": np.asarray(goal, dtype=np.float64),
            "body": int(scene.model.body("stove_main_group_door").id),
        }
        sol, _ = solve_arm(scene.model, scene.data, goal_spec, scene.qadr, scene.vadr)
        q_des = sol.astype(np.float32)
        q_des[:15] = STAND_Q[:15]
    tau = compute_torques(scene.model, scene.data, q_des, scene.kp, scene.kd, scene.qadr, scene.vadr)
    scene.data.ctrl[:N_ACT] = tau
    # Soft weld: force on the *robot* wrist body only (allowed under Rule 7).
    cur = np.asarray(scene.data.xpos[wrist_body], dtype=np.float64)
    scene.data.xfrc_applied[wrist_body, :3] = 120.0 * (goal - cur)
    scene.data.xfrc_applied[wrist_body, 3:] = 0.0
    return q_des


def run_ablation_episode(
    seed: int,
    stage: Stage,
    *,
    base: BaseMode = "frozen",
    seconds: float = 12.0,
    scene: OvenScene | None = None,
    scene_mod: str = "none",
) -> AblationResult:
    import time

    own = scene is None
    scene = scene or OvenScene()
    robot_bodies = robot_body_ids(scene.model)
    robot_dofs = robot_dof_ids(scene.model, robot_bodies)
    hand_gids = hand_geom_ids(scene.model, robot_bodies)
    handle_gid = int(scene.model.geom(HANDLE_GEOM).id)
    door_dof = scene._door_dof

    diagnostic = stage == "s1_weld"
    freeze = base == "frozen"
    t0 = time.perf_counter()
    applied = scene.reset(int(seed), randomize=True, theta=DEFAULT_THETA)
    # Clear any leftover applied forces.
    scene.data.qfrc_applied[:] = 0.0
    scene.data.xfrc_applied[:] = 0.0

    rule = Rule7State.capture(scene.model, scene.data, allow_equality=diagnostic)
    tracker = ContactOpenTracker(
        handle_gid=handle_gid,
        hand_gids=hand_gids,
        dt=scene.dt,
    )
    # Refine lever arm from geometry.
    priv0 = privileged_handle(scene.model, scene.data)
    arm = float(np.linalg.norm(priv0["handle_pos"] - priv0["hinge_pos"]))
    dparams = door_params(scene.model)
    dparams["arm_m_approx"] = arm

    invalid_reasons: list[str] = []
    max_angle = 0.0
    fell = False
    wrist_body = int(scene.model.body("right_wrist_yaw_link").id)

    if stage == "s1_weld":
        # Approach once, then arc-weld pull.
        teacher = HonestTeacher(scene, freeze_base=True, open_fingers=False)
        teacher.reset(DEFAULT_THETA)
        # Place at stance
        priv = privileged_handle(scene.model, scene.data)
        x, y, yaw = teacher._stance_pose(priv)
        scene._place_robot(x, y, yaw)
        scene._settle(0.2)
        handle0 = priv["handle_pos"].copy()
        hinge0 = priv["hinge_pos"].copy()
        axis0 = priv["hinge_axis"].copy()
        max_ticks = max(1, int(seconds / scene.dt))
        q_des = None
        for tick in range(max_ticks):
            frac = min(1.0, tick * scene.dt / 4.0)
            scene.data.qfrc_applied[:] = 0.0
            scene.data.xfrc_applied[:] = 0.0
            # Re-solve IK only every 0.2 s — weld force does the fine tracking.
            if tick % max(1, int(0.2 / scene.dt)) == 0:
                q_des = None
            q_des = _weld_wrist_to_handle(
                scene, wrist_body, handle0, hinge0, axis0, frac, q_des=q_des
            )
            # freeze base
            scene.data.qpos[0] = x
            scene.data.qpos[1] = y
            scene.data.qvel[0:6] = 0.0
            mujoco.mj_step(scene.model, scene.data)
            ang = oven_open_angle(scene.model, scene.data)
            max_angle = max(max_angle, ang)
            tracker.step(scene.model, scene.data, ang, door_dof)
            # Stage 1: robot xfrc on wrist is allowed; world must stay clean.
            for bid in range(scene.model.nbody):
                if bid in robot_bodies:
                    continue
                if float(np.linalg.norm(scene.data.xfrc_applied[bid])) > 1e-9:
                    invalid_reasons.append(f"xfrc_world_body={bid}")
            for did in range(scene.model.nv):
                if did in robot_dofs:
                    continue
                if abs(float(scene.data.qfrc_applied[did])) > 1e-9:
                    invalid_reasons.append(f"qfrc_world_dof={did}")
            if scene.fell():
                fell = True
                break
            if tracker.success():
                break
        # Stage 1 is diagnostic — success logged but cheat flagged; excluded from gates.
        success = tracker.success() and not fell and not invalid_reasons
        balance = "base_frozen_diagnostic"
    else:
        open_fingers = stage == "s2_hook"
        teacher = HonestTeacher(scene, freeze_base=freeze, open_fingers=open_fingers)
        teacher.reset(DEFAULT_THETA)
        max_ticks = max(1, int(seconds / scene.dt))
        for _ in range(max_ticks):
            scene.data.qfrc_applied[:] = 0.0
            # clear world xfrc; robot xfrc unused
            for bid in range(scene.model.nbody):
                if bid not in robot_bodies:
                    scene.data.xfrc_applied[bid] = 0.0
            phase = teacher.step()
            # Enforce Rule 7 after teacher step (teacher must not set door qfrc).
            scene.data.qfrc_applied[door_dof] = 0.0
            ang = oven_open_angle(scene.model, scene.data)
            max_angle = max(max_angle, float(ang), float(teacher.max_angle))
            tracker.step(scene.model, scene.data, float(ang), door_dof)
            for bid in range(scene.model.nbody):
                if bid in robot_bodies:
                    continue
                if float(np.linalg.norm(scene.data.xfrc_applied[bid])) > 1e-9:
                    invalid_reasons.append(f"xfrc_world_body={bid}")
            for did in range(scene.model.nv):
                if did in robot_dofs:
                    continue
                if abs(float(scene.data.qfrc_applied[did])) > 1e-9:
                    invalid_reasons.append(f"qfrc_world_dof={did}")
            if scene.fell():
                fell = True
                break
            if tracker.success():
                break
            if phase == Phase.DONE and tracker.hold_s <= 0.0 and teacher.max_angle < 0.05:
                # early exit if never engaged
                if teacher.phase == Phase.DONE and len(scene._angle_hist) > int(2.0 / scene.dt):
                    pass
        success = tracker.success() and not fell and not invalid_reasons
        balance = "base_frozen_qpos" if freeze else "joint_pd_stand_free_base"

    # Dedup invalid reasons
    invalid_reasons = sorted(set(invalid_reasons))
    invalid = bool(invalid_reasons)
    if invalid:
        success = False

    wall = time.perf_counter() - t0
    sim_s = len(scene._angle_hist) * scene.dt
    return AblationResult(
        seed=int(seed),
        stage=stage,
        base=base if stage == "s3_honest" else ("frozen" if stage != "s3_honest" else base),
        scene_mod=scene_mod,
        success=bool(success),
        fell=bool(fell),
        invalid_episode=invalid,
        invalid_reasons=invalid_reasons,
        diagnostic_cheat=diagnostic,
        max_angle=float(max_angle),
        sim_s=float(sim_s),
        wall_s=float(wall),
        contact=tracker.as_dict(),
        door=dparams,
        extras={
            "balance": balance,
            "prompt": applied.get("prompt"),
            "open_fingers": stage == "s2_hook",
        },
    )
