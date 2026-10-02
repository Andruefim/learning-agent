"""Oven-open task: physics success metric, randomization, privileged geometry.

Success is judged only by the door hinge angle — never by a VLM.
Student inputs must not include privileged fields from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np

from agent.config import FALL_Z
from agent.h2 import MODEL_XML, N_ACT, SPAWN_XY, SPAWN_Z, STAND_Q, actuator_addrs
from agent.joint_pd import compute_torques, kp_kd_vectors

DOOR_JOINT = "stove_main_group_door_joint"
DOOR_BODY = "stove_main_group_door"
HANDLE_GEOM = "stove_main_group_door_handle_main"
STOVE_BODY = "stove_main_group_main"
SUCCESS_ANGLE = 0.8  # rad
SUCCESS_HOLD_S = 1.0
EPISODE_S = 60.0
# Kitchen default stiffness lets gravity/contact fling the door open after a stove shift.
DOOR_DAMP_EVAL = 8.0
DOOR_FRIC_EVAL = 6.0
DOOR_STIFF_EVAL = 12.0
SETTLE_S = 0.4
PROMPTS = (
    "open the oven",
    "open the oven door",
    "open oven",
    "открой духовку",
    "открой дверцу духовки",
    "pull open the stove door",
)

DEFAULT_THETA = {
    "stance_x": -0.55,
    "stance_y": 0.0,
    "stance_yaw": 0.0,
    "contact_offset_along_handle": 0.0,
    "pregrasp_clearance": 0.08,
    "pull_speed": 0.25,
    "grip_close": 0.85,
    "pull_arc_scale": 1.0,
}

THETA_BOUNDS = {
    "stance_x": (-0.90, -0.35),
    "stance_y": (-0.25, 0.25),
    "stance_yaw": (-0.35, 0.35),
    "contact_offset_along_handle": (-0.25, 0.25),
    "pregrasp_clearance": (0.03, 0.18),
    "pull_speed": (0.08, 0.45),
    "grip_close": (0.40, 1.0),
    "pull_arc_scale": (0.6, 1.4),
}


@dataclass
class EpisodeResult:
    seed: int
    prompt: str
    theta: dict
    outcome: str  # success | timeout | fall
    max_angle: float
    success: bool
    fell: bool
    sim_s: float
    wall_s: float
    reason: str = ""
    angle_trace: list[float] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["theta"] = {k: float(v) for k, v in self.theta.items()}
        return d


def door_joint_id(model: mujoco.MjModel) -> int:
    return int(model.joint(DOOR_JOINT).id)


def door_qpos_adr(model: mujoco.MjModel) -> int:
    return int(model.jnt_qposadr[door_joint_id(model)])


def door_dof_adr(model: mujoco.MjModel) -> int:
    return int(model.jnt_dofadr[door_joint_id(model)])


def oven_open_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Hinge angle of the oven door in radians (0 closed → π/2 fully open)."""
    return float(data.qpos[door_qpos_adr(model)])


def is_success(history: list[float] | np.ndarray, *, dt: float, threshold: float = SUCCESS_ANGLE, hold_s: float = SUCCESS_HOLD_S) -> bool:
    """True when angle stays ≥ threshold for at least hold_s consecutive seconds."""
    if not len(history):
        return False
    need = max(1, int(np.ceil(hold_s / max(float(dt), 1e-6))))
    run = 0
    for a in history:
        if float(a) >= threshold:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


def clamp_theta(theta: dict) -> dict:
    out = dict(DEFAULT_THETA)
    out.update({k: float(v) for k, v in (theta or {}).items() if k in THETA_BOUNDS})
    for k, (lo, hi) in THETA_BOUNDS.items():
        out[k] = float(np.clip(out[k], lo, hi))
    return out


def privileged_handle(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """World-frame handle pose and door hinge axis — teacher / metric only."""
    gid = int(model.geom(HANDLE_GEOM).id)
    jid = door_joint_id(model)
    body = int(model.jnt_bodyid[jid])
    body_xpos = np.asarray(data.xpos[body], dtype=np.float64)
    body_xmat = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    hinge_local = np.asarray(model.jnt_pos[jid], dtype=np.float64)
    axis_local = np.asarray(model.jnt_axis[jid], dtype=np.float64)
    hinge_world = body_xpos + body_xmat @ hinge_local
    axis_world = body_xmat @ axis_local
    axis_world = axis_world / max(float(np.linalg.norm(axis_world)), 1e-8)
    handle_pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).copy()
    handle_mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3).copy()
    # Handle box long axis is local X (~0.69 m width).
    handle_along = handle_mat[:, 0]
    return {
        "handle_pos": handle_pos,
        "handle_mat": handle_mat,
        "handle_along": handle_along,
        "hinge_pos": hinge_world,
        "hinge_axis": axis_world,
        "door_angle": oven_open_angle(model, data),
        "stove_body": int(model.body(STOVE_BODY).id),
        "handle_geom": gid,
    }


class OvenScene:
    """MuJoCo kitchen + G1 with PD stand. No L3/VLA — for headless task eval."""

    def __init__(self, *, xml: str | None = None):
        path = str(xml or MODEL_XML)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.vadr = actuator_addrs(self.model)
        self.qadr = self.qadr[:N_ACT]
        self.vadr = self.vadr[:N_ACT]
        self.kp, self.kd = kp_kd_vectors()
        self.pelvis_id = int(self.model.body("pelvis").id)
        self.torso_id = int(self.model.body("torso_link").id)
        self._door_jid = door_joint_id(self.model)
        self._door_dof = door_dof_adr(self.model)
        self._door_qadr = door_qpos_adr(self.model)
        self._stove_bid = int(self.model.body(STOVE_BODY).id)
        self._stove_pos0 = np.asarray(self.model.body_pos[self._stove_bid], dtype=np.float64).copy()
        # Eval defaults (stable closed door). ±30% randomization applies on top.
        self.model.dof_damping[self._door_dof] = DOOR_DAMP_EVAL
        self.model.dof_frictionloss[self._door_dof] = DOOR_FRIC_EVAL
        self.model.jnt_stiffness[self._door_jid] = DOOR_STIFF_EVAL
        self._door_damp0 = float(self.model.dof_damping[self._door_dof])
        self._door_fric0 = float(self.model.dof_frictionloss[self._door_dof])
        self._door_stiff0 = float(self.model.jnt_stiffness[self._door_jid])
        self._geom_rgba0 = np.asarray(self.model.geom_rgba, dtype=np.float64).copy()
        self._light_diffuse0 = np.asarray(self.model.light_diffuse, dtype=np.float64).copy() if self.model.nlight else None
        self.prompt = PROMPTS[0]
        self.theta = dict(DEFAULT_THETA)
        self.seed = 0
        self._angle_hist: list[float] = []
        self._success_open_s = 0.0

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset_nominal(self) -> None:
        self.model.body_pos[self._stove_bid] = self._stove_pos0
        self.model.dof_damping[self._door_dof] = self._door_damp0
        self.model.dof_frictionloss[self._door_dof] = self._door_fric0
        self.model.jnt_stiffness[self._door_jid] = self._door_stiff0
        self.model.geom_rgba[:] = self._geom_rgba0
        if self._light_diffuse0 is not None:
            self.model.light_diffuse[:] = self._light_diffuse0

    def _lock_door_closed(self) -> None:
        self.data.qpos[self._door_qadr] = 0.0
        self.data.qvel[self._door_dof] = 0.0

    def _settle(self, seconds: float = SETTLE_S) -> None:
        """Resolve penetrations after stove/robot moves while holding the door shut."""
        ticks = max(1, int(seconds / self.dt))
        for _ in range(ticks):
            self._lock_door_closed()
            tau = compute_torques(self.model, self.data, STAND_Q, self.kp, self.kd, self.qadr, self.vadr)
            self.data.ctrl[:N_ACT] = tau
            if self.data.ctrl.size > N_ACT:
                self.data.ctrl[N_ACT:] = 0.0
            mujoco.mj_step(self.model, self.data)
        self._lock_door_closed()
        mujoco.mj_forward(self.model, self.data)

    def randomize(self, rng: np.random.Generator) -> dict:
        """Perturb spawn, stove, door friction, visuals, prompt. Returns applied dict."""
        self.reset_nominal()
        dx = float(rng.uniform(-0.30, 0.30))
        dy = float(rng.uniform(-0.30, 0.30))
        yaw = float(rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0)))
        # Do NOT translate stove body_pos: kitchen fixtures interpenetrate and the
        # neighboring cabinet door (stack_3) then wedges the oven door open.
        # Domain randomization for the stove is door dynamics + visual tint instead.
        stove_dx = 0.0
        stove_dy = 0.0
        damp_scale = float(rng.uniform(0.70, 1.30))
        fric_scale = float(rng.uniform(0.70, 1.30))
        stiff_scale = float(rng.uniform(0.70, 1.30))
        self.model.dof_damping[self._door_dof] = self._door_damp0 * damp_scale
        self.model.dof_frictionloss[self._door_dof] = self._door_fric0 * fric_scale
        self.model.jnt_stiffness[self._door_jid] = self._door_stiff0 * stiff_scale
        # Mild lighting / color jitter (visual only).
        if self.model.ngeom > 0:
            tint = float(rng.uniform(0.85, 1.10))
            self.model.geom_rgba[:, :3] = np.clip(self._geom_rgba0[:, :3] * tint, 0.0, 1.0)
        if self._light_diffuse0 is not None:
            self.model.light_diffuse[:] = np.clip(
                self._light_diffuse0 * float(rng.uniform(0.75, 1.25)), 0.0, 1.0
            )
        self.prompt = str(PROMPTS[int(rng.integers(0, len(PROMPTS)))])
        applied = {
            "spawn_dx": dx,
            "spawn_dy": dy,
            "spawn_yaw": yaw,
            "stove_dx": stove_dx,
            "stove_dy": stove_dy,
            "door_damp_scale": damp_scale,
            "door_fric_scale": fric_scale,
            "door_stiff_scale": stiff_scale,
            "prompt": self.prompt,
        }
        self._place_robot(SPAWN_XY[0] + dx, SPAWN_XY[1] + dy, yaw)
        self._settle()
        return applied

    def _place_robot(self, x: float, y: float, yaw: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (float(x), float(y), SPAWN_Z)
        self.data.qpos[3] = float(np.cos(yaw / 2.0))
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = float(np.sin(yaw / 2.0))
        for i in range(N_ACT):
            self.data.qpos[int(self.qadr[i])] = float(STAND_Q[i])
        self.data.qvel[:] = 0.0
        self._lock_door_closed()
        mujoco.mj_forward(self.model, self.data)
        self._angle_hist = []
        self._success_open_s = 0.0

    def reset(self, seed: int, *, randomize: bool = True, theta: dict | None = None) -> dict:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.theta = clamp_theta(theta or DEFAULT_THETA)
        if randomize:
            applied = self.randomize(rng)
        else:
            self.reset_nominal()
            self.prompt = PROMPTS[0]
            self._place_robot(float(SPAWN_XY[0]), float(SPAWN_XY[1]), 0.0)
            self._settle(0.2)
            applied = {"prompt": self.prompt}
        return applied

    def fell(self) -> bool:
        z = float(self.data.xpos[self.pelvis_id][2])
        up = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        return z < FALL_Z or up < 0.35

    def step_stand(self) -> float:
        """Zero policy: hold STAND_Q with joint PD. Returns door angle."""
        tau = compute_torques(self.model, self.data, STAND_Q, self.kp, self.kd, self.qadr, self.vadr)
        self.data.ctrl[:N_ACT] = tau
        if self.data.ctrl.size > N_ACT:
            self.data.ctrl[N_ACT:] = 0.0
        mujoco.mj_step(self.model, self.data)
        angle = oven_open_angle(self.model, self.data)
        self._angle_hist.append(angle)
        if angle >= SUCCESS_ANGLE:
            self._success_open_s += self.dt
        else:
            self._success_open_s = 0.0
        return angle

    def success_now(self) -> bool:
        return self._success_open_s >= SUCCESS_HOLD_S


def run_zero_policy_episode(
    seed: int,
    *,
    seconds: float = EPISODE_S,
    randomize: bool = True,
    scene: OvenScene | None = None,
) -> EpisodeResult:
    import time

    own = scene is None
    scene = scene or OvenScene()
    t0 = time.perf_counter()
    applied = scene.reset(seed, randomize=randomize)
    max_ticks = max(1, int(seconds / scene.dt))
    max_angle = 0.0
    outcome = "timeout"
    reason = "timeout"
    fell = False
    for _ in range(max_ticks):
        angle = scene.step_stand()
        max_angle = max(max_angle, angle)
        if scene.fell():
            fell = True
            outcome = "fall"
            reason = "fall"
            break
        if scene.success_now():
            outcome = "success"
            reason = "success"
            break
    wall = time.perf_counter() - t0
    sim_s = len(scene._angle_hist) * scene.dt
    # subsample angle trace for jsonl (every ~0.1 s)
    stride = max(1, int(0.1 / scene.dt))
    trace = [float(a) for a in scene._angle_hist[::stride]]
    result = EpisodeResult(
        seed=int(seed),
        prompt=str(applied.get("prompt") or scene.prompt),
        theta=dict(scene.theta),
        outcome=outcome,
        max_angle=float(max_angle),
        success=outcome == "success",
        fell=fell,
        sim_s=float(sim_s),
        wall_s=float(wall),
        reason=reason,
        angle_trace=trace,
        extras={"randomize": applied, "policy": "zero_stand"},
    )
    if own:
        pass
    return result
