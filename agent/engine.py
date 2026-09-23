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
    SKILL_TO_I,
    TRIAL_MAX,
    VISION_H,
    VISION_STRIDE,
    VISION_W,
)
from agent.flywheel import FlywheelMixin
from agent.g1_walk import load_g1_walk
from agent.h2 import (
    ACTION_DIM,
    KD,
    KP,
    MODEL_XML,
    N_ACT,
    SPAWN_Z,
    STAND_Q,
    actuator_addrs,
    arm_hang_cmd,
    box_geom,
    colliding_geoms,
    joint_limits,
)
from agent.joint_pd import compute_torques
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, CMD_VY, CMD_WZ, clip_command, command_from_step, stand_command
from agent.l3_foundation import (
    ACTION_FILTER_ALPHA,
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
from agent.plan import Plan, parse_requested_yaw, skill_from_params, wrap_angle
from agent.reach import attach_reach, reach_goal_from_step, servo_reach, solve_arm
from agent.planner import Level1Planner
from agent.policy import encode_instr, load_state, resolve_device
from agent.s1 import CTX, HIST_DIM, N_RAYS, CommandTransformer
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
            raise RuntimeError(f"G1 nu={self.model.nu}, expected {N_ACT}")
        self.renderer: mujoco.Renderer | None = None
        self.eye: mujoco.Renderer | None = None
        self.pelvis_id = self.model.body("pelvis").id
        self.torso_id = self.model.body("torso_link").id
        self.r_foot_id = self.model.body("right_ankle_roll_link").id
        self.l_foot_id = self.model.body("left_ankle_roll_link").id
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
        self.policy = CommandTransformer().to(self.device)
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
        self.walk = load_g1_walk()
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
        self._notes: list[str] = []
        self._sleep_jobs: list[list] = []
        self._sleeps = 0
        self._ep_i = 0
        self._ep_tick0 = 0
        self._ep_armed = False
        self._ep_slept = False
        self._ep_failed = False
        self._quiet_tick: int | None = None
        self._reach_sol: np.ndarray | None = None
        self._reach_tick = -10**9
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
        if self.walk is not None:
            self.walk.reset()
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
        self._reset_hist()
        self._rays = np.ones(N_RAYS, dtype=np.float32)
        self._load_queue(self.waypoint)
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
            self._finish_episode(train=False)
            self._ep_armed = False
            self._ep_slept = False
            self._ep_failed = False
            self._quiet_tick = None
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

    def _reset_hist(self) -> None:
        self._hist = np.zeros((CTX, HIST_DIM), dtype=np.float32)
        self._hist_n = 0

    def _push_hist(self, cmd: np.ndarray, err: np.ndarray, rays: np.ndarray) -> None:
        tok = np.zeros(HIST_DIM, dtype=np.float32)
        tok[:ACTION_DIM] = np.asarray(cmd, dtype=np.float32).reshape(ACTION_DIM)
        tok[ACTION_DIM : ACTION_DIM + 3] = np.asarray(err, dtype=np.float32).reshape(3)
        tok[ACTION_DIM + 3 :] = np.asarray(rays, dtype=np.float32).reshape(N_RAYS)
        if self._hist_n < CTX:
            self._hist[self._hist_n] = tok
            self._hist_n += 1
        else:
            self._hist[:-1] = self._hist[1:]
            self._hist[-1] = tok

    def _frame(self, step: dict) -> dict:
        step = step if isinstance(step, dict) else {}
        params = dict(step.get("params") or {})
        for key in ("direction", "speed", "depth", "pose", "hand", "hands", "distance_hint", "foot", "angle", "steps"):
            if key in step and key not in params:
                params[key] = step[key]
        skill = str(step.get("skill") or "").strip().lower()
        if skill not in SKILL_TO_I:
            hinted = dict(params)
            if "vx" in step:
                hinted["vx"] = float(step["vx"])
            skill = skill_from_params(hinted)
        mini = Plan(skill=skill, params=params)
        cmd = command_from_step(step, exec_bias=self._exec_bias)
        teacher = mini.teacher()
        yaw = parse_requested_yaw(mini.skill, mini.params)
        steps = int(teacher.steps)
        if "hold_s" in step:
            hold = float(step["hold_s"])
        elif steps > 0 or yaw is not None:
            hold = 30.0
        else:
            hold = 4.0
        return {
            "cmd": cmd,
            "hold_s": hold,
            "steps": steps,
            "yaw": yaw,
            "skill": mini.skill,
            "r_arm": float(teacher.r_arm),
            "l_arm": float(teacher.l_arm),
            "wave": float(teacher.wave),
            "kick": float(teacher.kick),
            "height": float(teacher.height),
            "intent_yaw": float(teacher.yaw),
            "reach": reach_goal_from_step(step),
        }

    def _servo_reach(self, cmd: np.ndarray) -> np.ndarray:
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = frame.get("reach") if frame else None
        if not goal or self.outcome == "fall":
            self._reach_sol = None
            return cmd
        if self._reach_sol is None or int(self._tick) - int(self._reach_tick) >= 300:
            self._reach_sol = solve_arm(self.model, self.data, goal, self.qadr, self.vadr)
            self._reach_tick = int(self._tick)
        return servo_reach(self.model, self.data, cmd, goal, self.qadr, self._reach_sol)

    def _load_queue(self, plan: Plan) -> None:
        attach_reach(plan, self.user_cmd or plan.instruction)
        self._reach_sol = None
        self._reach_tick = -10**9
        frames = [self._frame(step) for step in (plan.queue or [{"skill": "stand"}])]
        self._queue = frames or [self._frame({"skill": "stand"})]
        self._queue_i = 0
        self._arm_queue_step(0)

    def _arm_queue_step(self, index: int) -> None:
        frame = self._queue[index]
        self._steps_goal = int(frame["steps"])
        self._step_count = 0
        self._steps_done = 0
        self._walk_ticks = 0
        self._queue_tick0 = int(self._tick)
        if frame["yaw"] is not None:
            self._yaw_applied = 0.0
            self._turn_heading0 = self._heading()
            self._requested_yaw = float(frame["yaw"])
        else:
            self._turn_heading0 = None
            self._requested_yaw = None

    def _advance_queue(self) -> None:
        if not getattr(self, "_queue", None) or self._queue_i >= len(self._queue) - 1:
            return
        frame = self._queue[self._queue_i]
        elapsed = (int(self._tick) - int(self._queue_tick0)) * float(self.model.opt.timestep)
        done = elapsed >= float(frame["hold_s"])
        if int(frame["steps"]) > 0 and self._step_count >= int(frame["steps"]):
            done = True
        if frame["yaw"] is not None and self._turn_done():
            done = True
        if done:
            self._queue_i += 1
            self._arm_queue_step(self._queue_i)

    def _plan_command(self) -> np.ndarray:
        self._advance_queue()
        cmd = np.array(self._queue[self._queue_i]["cmd"], dtype=np.float32, copy=True)
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
        off = com_err_xy(
            self.data,
            self.pelvis_id,
            self.r_fg,
            self.l_fg,
            vx,
            self._gait_phi,
        )
        h = float(self._pelvis()[2]) - float(self._cmd[CMD_H])
        return np.array([off[0], off[1], h], dtype=np.float32)

    def scene_brief(self) -> dict:
        with self.lock:
            err = self._balance_err()
            frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
            cmd = frame["cmd"] if frame is not None else self._cmd
            return {
                "pelvis_z": round(float(self._pelvis()[2]), 3),
                "tilt": round(self._tilt_up(), 3),
                "err": round(float(np.linalg.norm(err)), 3),
                "outcome": self.outcome,
                "skill": frame["skill"] if frame is not None else self.waypoint.skill,
                "vx": round(float(cmd[CMD_VX]), 2),
                "r_arm": round(float(frame["r_arm"]) if frame is not None else self.waypoint.teacher().r_arm, 2),
                "l_arm": round(float(frame["l_arm"]) if frame is not None else self.waypoint.teacher().l_arm, 2),
                "yaw": round(float(frame["intent_yaw"]) if frame is not None else self.waypoint.teacher().yaw, 2),
                "requested_yaw": None if self._requested_yaw is None else round(float(self._requested_yaw), 3),
                "achieved_yaw": round(self._achieved_yaw(), 3),
                "done": bool(self.waypoint.done or self._turn_done()),
                "wave": round(float(frame["wave"]) if frame is not None else self.waypoint.teacher().wave, 2),
                "kick": round(float(frame["kick"]) if frame is not None else self.waypoint.teacher().kick, 2),
                "steps_left": max(0, self._steps_goal - self._step_count),
                "queue_i": int(getattr(self, "_queue_i", 0)),
                "queue_len": len(getattr(self, "_queue", []) or []),
                "ahead_m": round(float(self._rays[0]) * 8.0, 2),
            }

    def _needs_home(self) -> bool:
        return self.outcome == "fall" or self._tilt_up() < TILT_LIM or float(self._pelvis()[2]) < FALL_Z

    def begin_command(self, text: str):
        with self.lock:
            text = text.strip()
            if self._needs_home():
                self._ep_failed = True
                self._finish_episode(train=False)
                self._home()
            else:
                self._finish_episode(train=True)
            self._ep_armed = True
            self._ep_slept = False
            self._ep_failed = False
            self._quiet_tick = None
            self._ep_tick0 = int(self._tick)
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
                self._load_queue(self.waypoint)
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
            if fresh or not getattr(self, "_queue", None):
                self._load_queue(plan)
                self._quiet_tick = None
            elif len(plan.queue) == 1 and len(self._queue) == 1:
                frame = self._frame(plan.queue[0])
                had_yaw = self._queue[0]["yaw"] is not None
                self._queue[0] = frame
                self._steps_goal = int(frame["steps"])
                if frame["yaw"] is not None and not had_yaw:
                    self._yaw_applied = 0.0
                    self._turn_heading0 = self._heading()
                    self._requested_yaw = float(frame["yaw"])
                elif frame["yaw"] is not None:
                    self._requested_yaw = float(frame["yaw"])
                else:
                    self._turn_heading0 = None
                    self._requested_yaw = None
            self.status = plan.instruction[:80]
            if fresh or jumped:
                self._clear_errors()

    def l1_due(self) -> bool:
        with self.lock:
            if self.l1_busy or not self.user_cmd:
                return False
            if self._steps_goal > 0 and self._step_count < self._steps_goal:
                return False
            if getattr(self, "_queue", None) and self._queue_i < len(self._queue) - 1:
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
        teacher = self._servo_reach(self._plan_command())
        chosen = teacher
        self.ctrl_source = "l3"
        if self._tick % VISION_STRIDE == 0:
            self._eye_rgb = self._render_eye()
            self._rays = self._head_rays()
        student = self._last_student
        if self.outcome != "fall" and self._tick % VISION_STRIDE == 0:
            hist = self._hist[: max(self._hist_n, 0)]
            student = self._student(self._eye_rgb, proprio, language, z, errors, history=hist if self._hist_n else None)
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
                    "rays": self._rays.tolist(),
                }
            )
            if len(self.logs) > 4_000:
                drop = len(self.logs) - 3_000
                self.logs = self.logs[drop:]
                self._ep_i = max(0, int(self._ep_i) - drop)
        if self.outcome != "fall":
            chosen = self._mix(teacher, student)
            if self.alpha > 1e-8:
                self.ctrl_source = f"l3+l2-{self.stage}"
        self._last_teacher = teacher
        self._cmd = self._servo_reach(chosen)
        if self._tick % VISION_STRIDE == 0:
            err_now = self.errors[-1] if self.errors else np.zeros(3, dtype=np.float32)
            self._push_hist(chosen, err_now, self._rays)
        if self.outcome != "fall" and self._tick % DECIMATION == 0:
            obs = build_obs(
                self.data,
                self.torso_id,
                self._hinges(),
                self._qd(),
                self._last_a,
                self._cmd,
                self._gait_phi,
            )
            x = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw_a = self.l3.act(x)[0].detach().cpu().numpy().astype(np.float32)
                self._last_a += np.float32(ACTION_FILTER_ALPHA) * (raw_a - self._last_a)
        vx = float(self._cmd[CMD_VX])
        self._gait_phi = advance_gait_phi(self._gait_phi, vx, float(self.model.opt.timestep))
        off = com_err_xy(
            self.data,
            self.pelvis_id,
            self.r_fg,
            self.l_fg,
            vx,
            self._gait_phi,
        )
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
        cmd_v = np.array([self._cmd[CMD_VX], self._cmd[CMD_VY], self._cmd[CMD_WZ]], dtype=np.float32)
        # Arms forward move the mass past the toes. The stiff stand cannot
        # catch that; the walk net can, by stepping, so it stays in the loop.
        reaching = bool(getattr(self, "_queue", None) and self._queue[self._queue_i].get("reach"))
        arms_out = reaching or float(np.max(np.abs(self._cmd[CMD_ARMS] - arm_hang_cmd()))) > 0.45
        if self.walk is not None:
            leg = self.walk.leg_torque(
                self.data,
                self.qadr,
                self.vadr,
                cmd_v,
                self.kp,
                self.kd,
                float(self.model.opt.timestep),
                dynamic=arms_out,
            )
            if leg is not None:
                self.data.ctrl[:12] = leg
                self.ctrl_source = "g1-walk"
            else:
                self.ctrl_source = "g1-stand"
        mujoco.mj_step(self.model, self.data)
        if self.walk is not None and (self._tick + 1) % DECIMATION == 0 and not self.walk.holding:
            self.walk.update(self.data, self.qadr, self.vadr, cmd_v, dynamic=arms_out)
        elif self.walk is not None and self.walk.holding:
            self.walk.reset()
        if self._fell():
            if self.outcome != "fall":
                self._drop_failed_episode()
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
        self._consider_sleep()

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
        self.eye.update_scene(self.data, camera="head")
        return np.ascontiguousarray(self.eye.render())

    def eye_jpeg(self) -> bytes:
        with self.lock:
            rgb = self._eye_rgb if self.eye is not None and int(self._eye_rgb.sum()) > 0 else self._render_eye()
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=78)
            return buf.getvalue()

    def _head_rays(self) -> np.ndarray:
        """Five distances in front of the head camera, scaled to 0..1 over 8 m."""
        cam = int(self.model.camera("head").id)
        origin = np.asarray(self.data.cam_xpos[cam], dtype=np.float64)
        rot = np.asarray(self.data.cam_xmat[cam], dtype=np.float64).reshape(3, 3)
        forward = -rot[:, 2]
        right = rot[:, 0]
        up = rot[:, 1]
        dirs = (
            forward,
            forward + 0.45 * right,
            forward - 0.45 * right,
            forward + 0.30 * up,
            forward - 0.30 * up,
        )
        out = np.ones(N_RAYS, dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        for i, direction in enumerate(dirs):
            vec = direction / max(float(np.linalg.norm(direction)), 1e-8)
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                vec,
                None,
                1,
                int(self.pelvis_id),
                geomid,
            )
            if dist >= 0.0:
                out[i] = float(np.clip(dist / 8.0, 0.0, 1.0))
        return out

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
                "height": round(float(self._queue[self._queue_i]["height"]) if getattr(self, "_queue", None) else self.waypoint.teacher().height, 2),
                "vx": round(self._active_vx(), 2),
                "r_arm": round(float(self._queue[self._queue_i]["r_arm"]) if getattr(self, "_queue", None) else self.waypoint.teacher().r_arm, 2),
                "l_arm": round(float(self._queue[self._queue_i]["l_arm"]) if getattr(self, "_queue", None) else self.waypoint.teacher().l_arm, 2),
                "yaw": round(float(self._queue[self._queue_i]["intent_yaw"]) if getattr(self, "_queue", None) else self.waypoint.teacher().yaw, 2),
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
                "skill": self._queue[self._queue_i]["skill"] if getattr(self, "_queue", None) else self.waypoint.skill,
                "params": self.waypoint.params,
                "queue_i": int(getattr(self, "_queue_i", 0)),
                "queue_len": len(getattr(self, "_queue", []) or []),
                "trials": [t.as_public(i + 1) for i, t in enumerate(self.trials.items())],
            }

    def loop(self, stop: threading.Event):
        self._render()
        n = 0
        while not stop.is_set():
            job = None
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
                if self._sleep_jobs:
                    job = self._sleep_jobs.pop(0)
            if job is not None:
                try:
                    msg = self._train_rows(job)
                except Exception as exc:
                    msg = f"Сон прерван: {exc}"
                    self.policy.eval()
                with self.lock:
                    self.policy.eval()
                    self._note(msg)
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
