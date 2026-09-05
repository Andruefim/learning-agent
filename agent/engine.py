"""Robot runtime: MuJoCo H2, foundation policy, Joint-PD, telemetry."""

from __future__ import annotations

import collections
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import mujoco

from agent.config import (
    ERROR_LEN,
    FALL_Z as CFG_FALL_Z,
    H1_SPEC,
    L1_PERIOD,
    ROOT,
    SLEW,
    STAND_Z,
    TRIAL_MAX,
    VISION_H,
    VISION_STRIDE,
    VISION_W,
)
from agent.flywheel import FlywheelMixin
from agent.h2 import (
    ACTION_DIM,
    KD,
    KP,
    MODEL_XML,
    N_ACT,
    SPAWN_Z,
    STAND_Q,
    actuator_addrs,
    box_geom,
    colliding_geoms,
    disable_foot_spheres,
    joint_limits,
)
from agent.joint_pd import compute_torques
from agent.l3_cmd import CMD_H, CMD_VX, clip_command, command_from_plan, stand_command
from agent.l3_foundation import (
    DECIMATION,
    FALL_Z,
    TILT_LIM,
    HumanoidFoundationPolicy,
    balance_delta,
    body_xy,
    build_obs,
    com_err_xy,
    height_01,
    q_from_action,
    advance_gait_phi,
)
from agent.plan import Plan, parse_requested_yaw, wrap_angle
from agent.planner import Level1Planner
from agent.policy import FlowPolicy, encode_instr, load_state, resolve_device
from agent.trials import MultiTrialBuffer

STEP_PERIOD = 0.55


class RobotEngine(FlywheelMixin):
    def __init__(self):
        self.storage = Path(os.getenv("STORAGE_DIR", ROOT / "flywheel_data"))
        self.storage.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        if int(self.model.nu) != N_ACT:
            raise RuntimeError(f"H2 nu={self.model.nu}, expected {N_ACT}")
        self.renderer: mujoco.Renderer | None = None
        self.eye: mujoco.Renderer | None = None
        self.pelvis_id = self.model.body("pelvis").id
        self.torso_id = self.model.body("torso_link").id
        self.r_foot_id = self.model.body("right_ankle_pitch_link").id
        self.l_foot_id = self.model.body("left_ankle_pitch_link").id
        disable_foot_spheres(self.model, (self.r_foot_id, self.l_foot_id))
        self.r_foot_geoms = colliding_geoms(self.model, self.r_foot_id)
        self.l_foot_geoms = colliding_geoms(self.model, self.l_foot_id)
        self.r_fg = box_geom(self.model, self.r_foot_geoms, self.r_foot_geoms[0] if self.r_foot_geoms else 0)
        self.l_fg = box_geom(self.model, self.l_foot_geoms, self.l_foot_geoms[0] if self.l_foot_geoms else 0)
        self._off_prev = np.zeros(2, dtype=np.float32)
        self.qadr, self.vadr = actuator_addrs(self.model)
        self.lo, self.hi = joint_limits(self.model)
        self.tau_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.tau_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.kp = KP.copy()
        self.kd = KD.copy()
        self.policy = FlowPolicy().to(self.device)
        self.l3 = HumanoidFoundationPolicy(zero_out=True).to(self.device)
        self.l3.eval()
        self.planner = Level1Planner()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-3)
        self.ckpt = self.storage / "student.pt"
        self.l3_ckpt = self.storage / "l3_foundation.pt"
        self.replay_path = self.storage / "replay.npz"
        self.baked = False
        self.h1_pass = False
        self.h1_report: dict = dict(H1_SPEC)
        self.student_drive = False
        self.outcome = "ok"
        self.policy.eval()
        load_state(self.policy, self.ckpt, self.device)
        self.l3_drive = load_state(self.l3, self.l3_ckpt, self.device)
        self.lock = threading.RLock()
        self.errors: collections.deque[np.ndarray] = collections.deque(maxlen=ERROR_LEN)
        self.user_cmd = ""
        self.intent = ""
        self.waypoint: Plan = Plan.stand()
        self.goal: Plan | None = None
        self.trials = MultiTrialBuffer(maxlen=TRIAL_MAX)
        self._exec_bias: dict[str, float] = {}
        self._trial_busy = False
        self._measure: dict | None = None
        self._sgd_steps = 0
        self._foot_mu0 = float(self.model.geom_friction[self.r_fg, 0])
        self._all_foot_geoms = list(self.r_foot_geoms) + list(self.l_foot_geoms)
        self.alpha = 0.0
        self.alpha_working = 0.0
        self.stage = "A"
        self.shadow_ok_streak = 0
        self.shadow_mse = 0.0
        self.shadow_mse_ema: float | None = None
        self._ema_ok_tick0: int | None = None
        self._ema_ok_fall0 = 0.0
        self._replay_n = 0
        self._shadow_falls: collections.deque[int] = collections.deque(maxlen=64)
        self._cmd = stand_command()
        self._last_teacher = stand_command()
        self._last_student = stand_command()
        self._last_a = np.zeros(N_ACT, dtype=np.float32)
        self.q_cmd = STAND_Q.copy()
        self.last_l1 = 0.0
        self.l1_ok = True
        self.l1_busy = False
        self.logs: list[dict] = []
        self.ctrl_source = "l3"
        self._jpeg = b""
        self._eye_rgb = np.zeros((VISION_H, VISION_W, 3), dtype=np.uint8)
        self._tick = 0
        self._gait_phi = 0.0
        self._step_count = 0
        self._steps_done = 0
        self._steps_goal = 0
        self._walk_ticks = 0
        self._yaw_applied = 0.0
        self._turn_heading0: float | None = None
        self._requested_yaw: float | None = None
        self.turn_mode = "none"
        self.turn_mechanism = "foundation"
        self.status = "ready"
        self._kick_render = True
        self._home()
        blob = self._load_replay()
        if blob is not None:
            self._replay_n = int(len(blob["action"]))

    def _clear_errors(self):
        self.errors.clear()
        for _ in range(ERROR_LEN):
            self.errors.append(np.zeros(3, dtype=np.float32))

    def _teleport_spawn(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0:3] = (0.0, 0.0, SPAWN_Z)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qadr] = STAND_Q
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        if getattr(self.data, "qacc_warmstart", None) is not None:
            self.data.qacc_warmstart[:] = 0
        if int(self.model.na):
            self.data.act[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.q_cmd = STAND_Q.copy()
        self._cmd = stand_command()
        self._last_a = np.zeros(N_ACT, dtype=np.float32)
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self._pd_torque(self.q_cmd)

    def _home(self, *, keep_trials: bool = False, keep_intent: bool = False):
        self._teleport_spawn()
        if not keep_intent:
            self.user_cmd = ""
            self.intent = ""
            self.waypoint = Plan.stand()
            self.goal = None
        self.last_l1 = 0.0
        self.l1_ok = True
        self.l1_busy = False
        self._tick = 0
        self._gait_phi = 0.0
        self._step_count = 0
        self._steps_done = 0
        self._steps_goal = 0
        self._walk_ticks = 0
        self._yaw_applied = 0.0
        self._turn_heading0 = None
        self._requested_yaw = None
        self.turn_mode = "none"
        self.outcome = "ok"
        self._off_prev = np.zeros(2, dtype=np.float32)
        self._clear_errors()
        self.q_cmd = STAND_Q.copy()
        self._cmd = stand_command()
        self._last_teacher = stand_command()
        self._last_student = stand_command()
        self._last_a = np.zeros(N_ACT, dtype=np.float32)
        self._measure = None
        self._exec_bias = {}
        self.status = "stand"
        self._kick_render = True
        if not keep_trials:
            self.trials.clear()
        self._restore_feet()

    def _restore_feet(self):
        mu = float(getattr(self, "_foot_mu0", 1.0))
        for gid in getattr(self, "_all_foot_geoms", (self.r_fg, self.l_fg)):
            self.model.geom_friction[gid, 0] = mu

    def reset_sim(self):
        with self.lock:
            self._trial_busy = False
            self._home()
            self.status = "reset"

    def _hinges(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float32)

    def _qd(self) -> np.ndarray:
        return self.data.qvel[self.vadr].astype(np.float32)

    def _pelvis(self) -> np.ndarray:
        return self.data.xpos[self.pelvis_id].copy()

    def _tilt_up(self) -> float:
        return float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])

    def _com_xy(self) -> np.ndarray:
        return self.data.subtree_com[self.pelvis_id, :2].copy()

    def _feet_xy(self) -> np.ndarray:
        return 0.5 * (self.data.geom_xpos[self.r_fg, :2] + self.data.geom_xpos[self.l_fg, :2])

    def _plan_command(self) -> np.ndarray:
        cmd = command_from_plan(self.waypoint, exec_bias=self._exec_bias)
        if self._steps_goal > 0 and self._step_count >= self._steps_goal:
            cmd[CMD_VX] = 0.0
        if self.outcome == "fall":
            cmd[CMD_VX] = 0.0
            cmd[CMD_H] = STAND_Z
        return clip_command(cmd)

    def _active_vx(self) -> float:
        return float(self._cmd[CMD_VX])

    def _balance_err(self) -> np.ndarray:
        vx = self._active_vx()
        off = com_err_xy(self.data, self.pelvis_id, self.r_fg, self.l_fg, vx)
        h = float(self._pelvis()[2]) - float(self._cmd[CMD_H])
        return np.array([off[0], off[1], h], dtype=np.float32)

    def scene_brief(self) -> dict:
        with self.lock:
            err = self._balance_err()
            return {
                "pelvis_z": round(float(self._pelvis()[2]), 3),
                "tilt": round(self._tilt_up(), 3),
                "err": round(float(np.linalg.norm(err)), 3),
                "outcome": self.outcome,
                "skill": self.waypoint.skill,
                "vx": round(self._active_vx(), 2),
                "r_arm": round(self.waypoint.teacher().r_arm, 2),
                "l_arm": round(self.waypoint.teacher().l_arm, 2),
                "yaw": round(self.waypoint.teacher().yaw, 2),
                "requested_yaw": None if self._requested_yaw is None else round(float(self._requested_yaw), 3),
                "achieved_yaw": round(self._achieved_yaw(), 3),
                "done": bool(self.waypoint.done or self._turn_done()),
                "wave": round(self.waypoint.teacher().wave, 2),
                "kick": round(self.waypoint.teacher().kick, 2),
                "steps_left": max(0, self._steps_goal - self._step_count),
            }

    def _needs_home(self) -> bool:
        return self.outcome == "fall" or self._tilt_up() < TILT_LIM or float(self._pelvis()[2]) < FALL_Z

    def begin_command(self, text: str):
        with self.lock:
            text = text.strip()
            if self._needs_home():
                self._home()
            self.intent = text
            self.user_cmd = text
            self.l1_ok = True
            self.outcome = "ok"
            self._step_count = 0
            self._steps_done = 0
            self._steps_goal = 0
            self._walk_ticks = 0
            self._exec_bias = {}
            self._clear_errors()
            self._yaw_applied = 0.0
            self._turn_heading0 = None
            self._requested_yaw = None
            self.status = self.user_cmd or "stand"

    def apply_plan(self, plan: Plan, *, fresh: bool, l1_ok: bool = True):
        with self.lock:
            self.last_l1 = time.monotonic()
            self.l1_ok = bool(l1_ok)
            if self.outcome == "fall":
                self.user_cmd = ""
                self.status = "failed"
                return
            if plan.done:
                self.user_cmd = ""
                self.intent = ""
                self.status = "done"
                self.waypoint = Plan.stand("hold")
                return
            jumped = (
                abs(plan.teacher().height - self.waypoint.teacher().height) > 0.08
                or abs(plan.teacher().vx - self.waypoint.teacher().vx) > 0.15
            )
            if fresh or plan.teacher().steps != self.waypoint.teacher().steps:
                self._step_count = 0
                self._steps_done = 0
                self._steps_goal = int(plan.teacher().steps)
                self._walk_ticks = 0
            if fresh:
                req = parse_requested_yaw(plan.skill, plan.params)
                same_goal = (
                    req is not None
                    and self._requested_yaw is not None
                    and abs(float(req) - float(self._requested_yaw)) < 1e-6
                    and self._turn_heading0 is not None
                )
                if not same_goal:
                    self._yaw_applied = 0.0
                    self._turn_heading0 = self._heading()
                    self._requested_yaw = req
                self.goal = plan
            self.waypoint = plan
            self.status = plan.instruction[:80]
            if fresh or jumped:
                self._clear_errors()

    def l1_due(self) -> bool:
        with self.lock:
            if self.l1_busy or not self.user_cmd:
                return False
            if self._steps_goal > 0 and self._step_count < self._steps_goal:
                return False
            return (time.monotonic() - self.last_l1) >= L1_PERIOD

    def begin_l1(self) -> bool:
        with self.lock:
            if self.l1_busy:
                return False
            self.l1_busy = True
            return True

    def end_l1(self):
        with self.lock:
            self.l1_busy = False

    def _slew(self, desired: np.ndarray) -> np.ndarray:
        q = self.q_cmd.astype(np.float32)
        return np.clip(q + np.clip(desired.astype(np.float32) - q, -SLEW, SLEW), self.lo, self.hi)

    def _pd_torque(self, q_cmd: np.ndarray) -> np.ndarray:
        return compute_torques(self.model, self.data, q_cmd, self.kp, self.kd, self.qadr, self.vadr)

    def _update_steps(self) -> None:
        if abs(self._active_vx()) > 0.08:
            self._walk_ticks += 1
            dt = float(self.model.opt.timestep)
            self._step_count = int(self._walk_ticks * dt / STEP_PERIOD)
            self._steps_done = self._step_count
        if self._turn_heading0 is not None:
            self._yaw_applied = self._achieved_yaw()

    def step(self):
        err = self._balance_err()
        self.errors.append(err.copy())
        proprio = self._hinges()
        language = encode_instr(self.waypoint.param_text())
        z = self.waypoint.z()
        errors = np.stack(self.errors)
        teacher = self._plan_command()
        chosen = teacher
        self.ctrl_source = "l3"
        if self._tick % VISION_STRIDE == 0:
            self._eye_rgb = self._render_eye()
        student = self._last_student
        if self.outcome != "fall" and self._tick % VISION_STRIDE == 0:
            student = self._student(self._eye_rgb, proprio, language, z, errors)
            self._last_student = student
            mse = float(np.mean((student - teacher) ** 2))
            self.shadow_mse = mse
            self._maybe_update_authority(mse, fell=self._fell())
            self.logs.append(
                {
                    "image": self._eye_rgb.copy(),
                    "proprio": proprio.tolist(),
                    "language": language.tolist(),
                    "z": z.tolist(),
                    "errors": errors.tolist(),
                    "action": teacher.tolist(),
                    "student": student.tolist(),
                    "shadow_mse": mse,
                    "alpha": float(self.alpha),
                    "stage": self.stage,
                    "skill": self.waypoint.skill,
                    "outcome": self.outcome,
                }
            )
            if len(self.logs) > 4_000:
                self.logs = self.logs[-3_000:]
        if self.outcome != "fall":
            chosen = self._mix(teacher, student)
            if self.alpha > 1e-8:
                self.ctrl_source = f"l3+l2-{self.stage}"
        self._last_teacher = teacher
        self._cmd = chosen
        if self.outcome != "fall" and self._tick % DECIMATION == 0:
            obs = build_obs(self.data, self.torso_id, self._hinges(), self._qd(), self._last_a, self._cmd)
            x = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                self._last_a = self.l3.act(x)[0].detach().cpu().numpy().astype(np.float32)
        vx = float(self._cmd[CMD_VX])
        self._gait_phi = advance_gait_phi(self._gait_phi, vx, float(self.model.opt.timestep))
        off = com_err_xy(self.data, self.pelvis_id, self.r_fg, self.l_fg, vx)
        d_off = off - self._off_prev
        self._off_prev = off.copy()
        yaw = self._heading()
        q_des = q_from_action(self._cmd, self._last_a, self._gait_phi) + balance_delta(
            body_xy(off, yaw),
            body_xy(d_off, yaw),
            height_01=height_01(float(self._cmd[CMD_H])),
            vx=vx,
        )
        q_des = np.clip(q_des, self.lo, self.hi)
        self.q_cmd = self._slew(q_des)
        self.data.ctrl[:] = self._pd_torque(self.q_cmd)
        mujoco.mj_step(self.model, self.data)
        if self._fell():
            self.outcome = "fall"
        elif (
            self.outcome == "fall"
            and self.status != "failed"
            and self._tilt_up() > 0.85
            and float(self._pelvis()[2]) > STAND_Z - 0.08
        ):
            self.outcome = "ok"
        if self._measure is not None:
            self._poll_measure()
        self._update_steps()
        self._tick += 1

    def _render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data, camera="demo")
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=78)
        self._jpeg = buf.getvalue()

    def _render_eye(self) -> np.ndarray:
        if self.eye is None:
            self.eye = mujoco.Renderer(self.model, VISION_H, VISION_W)
        self.eye.update_scene(self.data, camera="demo")
        return np.ascontiguousarray(self.eye.render())

    def jpeg(self) -> bytes:
        with self.lock:
            return self._jpeg

    def _heading(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _achieved_yaw(self) -> float:
        if self._turn_heading0 is None:
            return 0.0
        return wrap_angle(self._heading() - float(self._turn_heading0))

    def _turn_done(self) -> bool:
        req = self._requested_yaw
        if req is None:
            return False
        from agent.config import PARAM_OK

        return abs(float(req) - self._achieved_yaw()) <= PARAM_OK["yaw"]

    def _fell(self) -> bool:
        return self._tilt_up() < TILT_LIM or float(self._pelvis()[2]) < min(FALL_Z, CFG_FALL_Z)

    def telemetry(self) -> dict:
        with self.lock:
            p = self._pelvis()
            return {
                "type": "telemetry",
                "instruction": self.waypoint.instruction,
                "command": self.user_cmd,
                "subgoal": self.waypoint.instruction[:28],
                "plan": [self.user_cmd] if self.user_cmd else [],
                "holding": False,
                "error": round(float(np.linalg.norm(self._balance_err())), 4),
                "pelvis_z": round(float(p[2]), 3),
                "x": round(float(p[0]), 3),
                "tilt": round(self._tilt_up(), 3),
                "logs": len(self.logs),
                "device": str(self.device),
                "baked": self.baked,
                "h1": self.h1_pass,
                "h1_spec": H1_SPEC,
                "drive": self.student_drive,
                "outcome": self.outcome,
                "ctrl": self.ctrl_source,
                "stage": self.stage,
                "alpha": round(float(self.alpha), 3),
                "shadow_mse": round(float(self.shadow_mse), 4),
                "shadow_mse_ema": None if self.shadow_mse_ema is None else round(float(self.shadow_mse_ema), 4),
                "fall_rate": round(self._fall_rate(), 3),
                "replay": int(self._replay_count()),
                "height": round(self.waypoint.teacher().height, 2),
                "vx": round(self._active_vx(), 2),
                "r_arm": round(self.waypoint.teacher().r_arm, 2),
                "l_arm": round(self.waypoint.teacher().l_arm, 2),
                "yaw": round(self.waypoint.teacher().yaw, 2),
                "yaw_applied": round(float(self._yaw_applied), 3),
                "requested_yaw": None if self._requested_yaw is None else round(float(self._requested_yaw), 3),
                "achieved_yaw": round(self._achieved_yaw(), 3),
                "done": bool(self.waypoint.done or self._turn_done()),
                "turn_mode": self.turn_mode,
                "turn_mechanism": self.turn_mechanism,
                "wave": round(self.waypoint.teacher().wave, 2),
                "kick": round(self.waypoint.teacher().kick, 2),
                "steps_left": max(0, self._steps_goal - self._step_count),
                "l1_ok": self.l1_ok,
                "l1_url": self.planner.base_url,
                "l1_err": self.planner.last_err,
                "status": self.status,
                "skill": self.waypoint.skill,
                "params": self.waypoint.params,
                "trials": [t.as_public(i + 1) for i, t in enumerate(self.trials.items())],
            }

    def loop(self, stop: threading.Event):
        self._render()
        n = 0
        while not stop.is_set():
            with self.lock:
                if not self._trial_busy:
                    self.step()
                    n += 1
                    if n % 8 == 0 or self._kick_render:
                        self._kick_render = False
                        try:
                            self._render()
                        except Exception:
                            pass
            stop.wait(self.model.opt.timestep)

    def close(self):
        for attr in ("renderer", "eye"):
            r = getattr(self, attr)
            if r is None:
                continue
            try:
                r.close()
            except Exception:
                pass
            setattr(self, attr, None)
