"""Robot runtime: MuJoCo H2, foundation policy, Joint-PD, telemetry."""

from __future__ import annotations

import collections
import ctypes
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
from agent.flip_policy import wants_flip
from agent.flywheel import FlywheelMixin
from agent.g1_walk import load_g1_walk
from agent.h2 import (
    ACTION_DIM,
    KD,
    KP,
    MODEL_XML,
    N_ACT,
    SPAWN_Z,
    SPAWN_XY,
    STAND_Q,
    WAIST_P,
    WAIST_R,
    WAIST_Y,
    actuator_addrs,
    arm_hang_cmd,
    box_geom,
    colliding_geoms,
    disable_wrist_mesh_contacts,
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
from agent.dex3 import FINGER_KD, FINGER_KP, finger_target
from agent.plan import Plan, attach_arm_channel, parse_requested_yaw, skill_from_params, wrap_angle
from agent.reach import (
    LOOK_H,
    LOOK_W,
    CONTACT_S,
    GRASP_S,
    GRASP_TRIES,
    LIFT_M,
    LIFT_MPS,
    RAISE_GAP_M,
    STANDOFF_M,
    TOUCH_M,
    assembly_root,
    attach_reach,
    body_can_move,
    cast_grasp_uv,
    camera_basis,
    look_errors,
    goal_distance,
    hands_holding,
    palm_near_point,
    ray_direction,
    ray_hit,
    reach_goal_from_step,
    refine_arm,
    servo_reach,
    solve_arm,
    touch_lesson,
    velocity_toward,
)
from agent.planner import Level1Planner
from agent.policy import encode_instr, load_matching, load_state, resolve_device, resolve_pi0_device
from agent.s1 import CTX, HIST_DIM, N_RAYS, CommandTransformer
from agent.e2e import E2E_BALANCE, E2ETeacher, e2e_enabled, resolve_e2e_device
from agent.vla import VLA_H, VLA_W, TORCH_GATE, PI0_WAIT_S, arm_contact_geoms, pose_moved, subtree_bodies, System15
from agent.trials import MultiTrialBuffer

STEP_PERIOD = 0.55


# Operator preview. The head camera stays at VISION_H x VISION_W.
PREVIEW_W, PREVIEW_H = 1920, 1080
PREVIEW_INTERVAL = 1.0 / 40.0


class RobotEngine(FlywheelMixin):
    def __init__(self):
        self.storage = Path(os.getenv("STORAGE_DIR", ROOT / "flywheel_data"))
        self.storage.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        if int(self.model.nu) < N_ACT:
            raise RuntimeError(f"G1 nu={self.model.nu}, expected at least {N_ACT}")
        if self.model.actuator(N_ACT - 1).name != "right_wrist_yaw":
            raise RuntimeError("body actuators must stay the first 29, fingers after them")
        disable_wrist_mesh_contacts(self.model)
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
        qadr, vadr = actuator_addrs(self.model)
        lo, hi = joint_limits(self.model)
        self.qadr, self.vadr = qadr[:N_ACT], vadr[:N_ACT]
        self.lo, self.hi = lo[:N_ACT], hi[:N_ACT]
        self.finger_qadr, self.finger_vadr = qadr[N_ACT:], vadr[N_ACT:]
        self.finger_tau_lo = self.model.actuator_ctrlrange[N_ACT:, 0].astype(np.float32)
        self.finger_tau_hi = self.model.actuator_ctrlrange[N_ACT:, 1].astype(np.float32)
        self._finger_q = np.zeros(int(self.finger_qadr.size), dtype=np.float32)
        self.tau_lo = self.model.actuator_ctrlrange[:N_ACT, 0].astype(np.float32)
        self.tau_hi = self.model.actuator_ctrlrange[:N_ACT, 1].astype(np.float32)
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
        load_matching(self.policy, self.ckpt, self.device)
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
        self._awaiting_plan = False
        self._flip = None
        self._quiet_tick: int | None = None
        self._reach_sol: np.ndarray | None = None
        self._reach_tick = -10**9
        self._touch_goal = False
        self._touch_d0: float | None = None
        self._touch_best: float | None = None
        self._blocked_tick0: int | None = None
        self._approach_arrived = False
        self._reach_close = False
        self._relook_used = False
        self._walk_released = False
        self._arm_wait0: int | None = None
        self._gaze_locked = False
        self._need_gaze_plan = False
        self._jpeg_was_gaze = False
        self._marker_pos: np.ndarray | None = None
        self._marker_aim: np.ndarray | None = None
        self._marker_dim = False
        self._anchor: dict | None = None
        self._pi0_prefetch = False
        self._pi0_approach_hold = False
        self._verify_at = 0.0
        self._need_verify = False
        self._world_moved = False
        self._grasp_tick0: int | None = None
        self._slip_tick0: int | None = None
        self._contact_tick0: int | None = None
        self._grasp_mark = -1
        self._grasp_tries = 0
        self._shift_xy0 = None
        self._shift_key = None
        self.look = None
        # Demo camera orbits the pelvis. Offset is the current XML camera
        # relative to a standing pelvis. Group 0 is collision (solid red).
        self._cam_offset = np.array([-1.15, -1.70, 0.767], dtype=np.float64)
        self._view_opt = mujoco.MjvOption()
        self._view_opt.geomgroup[:] = (0, 1, 1, 0, 0, 0)
        self.ctrl_source = "l3"
        self._jpeg = b""
        self._eye_jpeg = b""
        self._look_jpeg = b""
        self._look_rgb: np.ndarray | None = None
        self._look_seq = 0
        self._look_jpeg_seq = -1
        self._eye_rgb = np.zeros((VISION_H, VISION_W, 3), dtype=np.uint8)
        self._eye_seq = 0
        self._eye_jpeg_seq = -1
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
        self._preview_at = 0.0
        self._preview_rgb: np.ndarray | None = None
        self._preview_seq = 0
        self._jpeg_seq = -1
        self._robot_bodies = subtree_bodies(self.model, int(self.pelvis_id))
        self._arm_geoms = arm_contact_geoms(self.model)
        self._e2e_mode = e2e_enabled()
        self.e2e: E2ETeacher | None = E2ETeacher(resolve_e2e_device()) if self._e2e_mode else None
        # Arms-only π0 stays available only when e2e is off (one GPU resident).
        self.vla = System15(resolve_pi0_device() if not self._e2e_mode else torch.device("cpu"))
        self._ep_pi0 = False
        self._ep_e2e = False
        self._e2e_driving = False
        self._e2e_prefetch = False
        self._e2e_q: np.ndarray | None = None
        self._pi0_driving = False
        self._pi0_touched: set[int] = set()
        self._pi0_x0 = None
        self._pi0_m0 = None
        self._pi0_wait0: int | None = None
        self._vla_eye = None
        self._home()
        blob = self._load_replay()
        if blob is not None:
            self._replay_n = int(len(blob["action"]))
        if self._e2e_mode:
            print(
                f"E2E ON  device={self.e2e.device}  L2={self.device}  "
                f"(выгрузи Ollama — на GPU только e2e)",
                flush=True,
            )
        else:
            print(f"L2 device={self.device}  π0 device={self.vla.device}", flush=True)
    def _clear_errors(self):
        self.errors.clear()
        for _ in range(ERROR_LEN):
            self.errors.append(np.zeros(3, dtype=np.float32))

    def _teleport_spawn(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0:3] = (float(SPAWN_XY[0]), float(SPAWN_XY[1]), SPAWN_Z)
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
        self._finger_q = np.zeros(int(self.finger_qadr.size), dtype=np.float32)
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        if self.walk is not None:
            self.walk.reset()
        mujoco.mj_forward(self.model, self.data)
        self._write_ctrl()

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
        self._awaiting_plan = False
        self._approach_arrived = False
        self._reach_close = False
        self._relook_used = False
        self._walk_released = False
        self._arm_wait0 = None
        self._gaze_locked = False
        self._need_gaze_plan = False
        self._jpeg_was_gaze = False
        self._marker_pos = None
        self._marker_aim = None
        self._marker_dim = False
        self._anchor = None
        self._pi0_prefetch = False
        self._e2e_prefetch = False
        self._e2e_driving = False
        self._e2e_q = None
        self._pi0_approach_hold = False
        self._verify_at = 0.0
        self._need_verify = False
        self._verify_hold_noted = False
        self._world_moved = False
        self._wall_cast_tries = 0
        self._reset_pi0_episode()
        self._reset_e2e_episode()
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

    def _note_touch(self, goal: dict) -> None:
        dist = goal_distance(self.model, self.data, goal)
        if dist is None:
            return
        self._touch_goal = True
        if self._touch_d0 is None:
            self._touch_d0 = float(dist)
        self._touch_best = float(dist) if self._touch_best is None else min(float(self._touch_best), float(dist))

    def _workspace_pitch(self) -> float:
        """Legacy lean toward a low counter. Unused for routine pointing — it hid tall objects."""
        jid = int(self.model.actuator_trnid[WAIST_P, 0])
        lo, hi = (float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1]))
        return float(np.clip(0.36, lo, hi))

    def _look_at_waist(self, point: np.ndarray) -> tuple[float, float]:
        """Waist yaw/pitch that centers the head camera on a world point."""
        err = look_errors(self.model, self.data, point)
        q = self.data.qpos
        adr = self.qadr
        cur_y = float(q[adr[WAIST_Y]])
        cur_p = float(q[adr[WAIST_P]])
        if err is None:
            return cur_y, cur_p
        yaw_err, pitch_err = err
        # Joint sense (calibrated): desired = current - camera_error.
        return (
            float(np.clip(cur_y - yaw_err, self.lo[WAIST_Y], self.hi[WAIST_Y])),
            float(np.clip(cur_p - pitch_err, self.lo[WAIST_P], self.hi[WAIST_P])),
        )

    def _should_look_at_goal(self, goal: dict | None) -> bool:
        """After the world point is known, watch it — except while freezing an upright relook frame."""
        if not goal or goal.get("point") is None or goal.get("target") == "head":
            return False
        if goal.get("stale") or goal.get("done"):
            return False
        if self._holding_gaze():
            return False
        if getattr(self, "_need_gaze_plan", False):
            return False
        # Keep upright until the approach walk is released, so the first cast stays consistent.
        if not getattr(self, "_walk_released", False):
            return False
        return True

    def _gaze_ready(self) -> bool:
        """Upright head frame for a relook JPEG. No mandatory lean into the floor."""
        if not self._pi0_task():
            return False
        q = self.data.qpos
        adr = self.qadr
        upright = abs(float(q[adr[WAIST_P]])) < 0.12
        straight = abs(float(q[adr[WAIST_Y]])) < 0.10 and abs(float(q[adr[WAIST_R]])) < 0.10
        return upright and straight

    def _holding_gaze(self) -> bool:
        """Stand still for a relook JPEG. Do not lean the waist."""
        if not self._pi0_task():
            return False
        if getattr(self, "_need_gaze_plan", False) and not getattr(self, "_gaze_locked", False):
            return True
        if not getattr(self, "_gaze_locked", False):
            return False
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        reach = (frame or {}).get("reach") or {}
        return "u" in reach and reach.get("point") is None and not reach.get("stale")

    def reach_prior(self) -> dict | None:
        """Pixel/world memory for a close relook — so L1 is not amnesic."""
        with self.lock:
            frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
            reach = (frame or {}).get("reach") or {}
            anchor = getattr(self, "_anchor", None) or {}
            u = reach.get("u", anchor.get("u"))
            v = reach.get("v", anchor.get("v"))
            if u is None or v is None:
                return None
            return {
                "u": float(u),
                "v": float(v),
                "hand": str(reach.get("hand") or anchor.get("hand") or "right"),
                "has_world": bool(anchor.get("point") is not None or reach.get("point") is not None),
            }

    def _body_name(self, body: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or "?"

    def _cast_is_junk(self, name: str, u: float, *, refining: bool = False) -> bool:
        low = name.lower()
        if "wall" in low or "floor" in low or low in {"world", "ground"}:
            return True
        # Border pixels are fine for a first approach cast, but not as a close refine.
        if refining and (u < 0.10 or u > 0.90):
            return True
        return False

    def _save_anchor(self, goal: dict) -> None:
        point = goal.get("point")
        body = goal.get("body")
        if point is None or body is None or "u" not in goal:
            return
        name = self._body_name(int(body))
        u = float(goal["u"])
        if self._cast_is_junk(name, u):
            return
        # Prefer not to remember an image-border grip as the lasting target.
        if u < 0.12 or u > 0.88:
            return
        self._anchor = {
            "point": np.asarray(point, dtype=np.float64).copy(),
            "aim": None if goal.get("aim") is None else np.asarray(goal["aim"], dtype=np.float64).copy(),
            "body": int(body),
            "u": u,
            "v": float(goal["v"]),
            "hand": str(goal.get("hand") or "right"),
            "name": name,
        }

    def _restore_anchor(self, goal: dict) -> bool:
        anchor = getattr(self, "_anchor", None)
        if not anchor or anchor.get("point") is None:
            return False
        goal["point"] = np.asarray(anchor["point"], dtype=np.float64).copy()
        if anchor.get("aim") is not None:
            goal["aim"] = np.asarray(anchor["aim"], dtype=np.float64).copy()
        goal["body"] = int(anchor["body"])
        goal["u"] = float(anchor["u"])
        goal["v"] = float(anchor["v"])
        goal["hand"] = str(anchor.get("hand") or goal.get("hand") or "right")
        goal.pop("stale", None)
        goal["drawn"] = 0.0
        self._marker_pos = np.asarray(goal["point"], dtype=np.float64).copy()
        self._marker_aim = None if goal.get("aim") is None else np.asarray(goal["aim"], dtype=np.float64).copy()
        self._marker_dim = False
        self._gaze_locked = True
        self._need_gaze_plan = False
        self._note(f"Держу цель с подхода: {anchor.get('name', '?')}")
        return True

    def _cast_look(self, goal: dict) -> None:
        if goal.get("stale") or goal.get("point") is not None or "u" not in goal:
            return
        # Pixel came from an upright head frame (first plan or relook). Wait out any old lean.
        if not getattr(self, "_gaze_locked", False):
            return
        if abs(float(self.data.qpos[self.qadr[WAIST_P]])) > 0.12:
            return
        origin, rot, fovy = camera_basis(self.model, self.data)
        direction = ray_direction(rot, fovy, LOOK_W / LOOK_H, float(goal["u"]), float(goal["v"]))
        hit = ray_hit(self.model, self.data, origin, direction, int(self.pelvis_id))
        if hit is None:
            if self._restore_anchor(goal):
                return
            self._note("Луч мимо. Нужен другой ракурс.")
            self._gaze_locked = False
            self._need_gaze_plan = True
            return
        point, body = hit
        root = int(assembly_root(self.model, int(body)))
        # Trust the planner pixel — no geom-name snap heuristics.
        chosen = np.asarray(point, dtype=np.float64)
        chosen_dir = direction
        name = self._body_name(root)
        refining = bool(getattr(self, "_anchor", None) or getattr(self, "_relook_used", False))
        if self._cast_is_junk(name, float(goal["u"]), refining=refining) and self._restore_anchor(goal):
            print(f"cast reject u={float(goal['u']):.2f} v={float(goal['v']):.2f} {name} -> anchor", flush=True)
            return
        if "wall" in name.lower() or "floor" in name.lower():
            # Same pose + new L1 frame just hits the wall again (see reconnect spam).
            if self._restore_anchor(goal):
                print(
                    f"cast wall/floor u={float(goal['u']):.2f} v={float(goal['v']):.2f} {name} -> anchor",
                    flush=True,
                )
                return
            tries = int(getattr(self, "_wall_cast_tries", 0)) + 1
            self._wall_cast_tries = tries
            if tries <= 1:
                self._note(f"Луч в {name}. Нужен другой ракурс.")
                self._gaze_locked = False
                self._need_gaze_plan = True
            else:
                self._need_gaze_plan = False
                self._gaze_locked = False
                self._note(f"Луч в {name}. Стоп переприцел — без якоря не кручу L1.")
            return
        self._wall_cast_tries = 0
        goal["body"] = root
        goal["aim"] = np.asarray(chosen_dir, dtype=np.float64).copy()
        goal["drawn"] = 0.0
        goal["point"] = chosen - chosen_dir * STANDOFF_M
        point = chosen
        self._marker_pos = np.asarray(goal["point"], dtype=np.float64).copy()
        self._marker_aim = np.asarray(chosen_dir, dtype=np.float64).copy()
        self._marker_dim = False
        self._save_anchor(goal)
        waist = float(self.data.qpos[self.qadr[WAIST_P]])
        print(
            f"cast u={float(goal['u']):.2f} v={float(goal['v']):.2f} {name} z={float(point[2]):.2f} waist={waist:.2f} yaw={self._heading():.2f} wy={float(self.data.qpos[self.qadr[WAIST_Y]]):.2f}",
            flush=True,
        )
        self._note(f"Цель: {name} · z={float(point[2]):.2f}")

    def _draw_contact(self, goal: dict) -> None:
        """Once the hand is holding, ease the point back along the look.

        Do not yank on first touch — that tips a planted stand into the fridge.
        """
        aim = goal.get("aim")
        point = goal.get("point")
        if aim is None or point is None:
            return
        if not goal.get("held"):
            return
        if float(getattr(self, "_reach_gap", 1.0)) > TOUCH_M * 1.5 and not self._reach_close:
            return
        drawn = float(goal.get("drawn") or 0.0)
        limit = 0.14
        if drawn >= limit:
            return
        # ~3 cm/s — was 0.12 m/s which ripped the torso forward on contact.
        step = min(limit - drawn, 0.03 * float(self.model.opt.timestep))
        goal["point"] = np.asarray(point, dtype=np.float64) - np.asarray(aim, dtype=np.float64) * step
        goal["drawn"] = drawn + step

    def _drop_grasp(self, goal: dict, frame: dict, note: str) -> None:
        self._grasp_tries += 1
        goal["held"] = False
        goal.pop("drawn", None)
        goal.pop("x0", None)
        goal.pop("z0", None)
        goal.pop("lift0", None)
        goal.pop("world_moved", None)
        self._marker_dim = True
        self._reach_close = False
        self._reach_sol = None
        self._reach_tick = -10**9
        self._blocked_tick0 = None
        self._grasp_tick0 = None
        self._slip_tick0 = None
        self._contact_tick0 = None
        self._shift_xy0 = None
        self._shift_key = None
        frame["approaching"] = False
        # Keep the plant. Clearing approach_arrived made him stomp away after a miss.
        if getattr(self, "_anchor", None) and self._restore_anchor(goal):
            self._approach_arrived = True
            self._walk_released = True
            self._need_gaze_plan = False
            self._gaze_locked = True
            goal.pop("stale", None)
            if self._grasp_tries >= GRASP_TRIES:
                goal["done"] = True
                self._note("Хват не получился.")
            else:
                self._note(note or "Хват мимо. Остаюсь у якоря.")
            return
        goal["stale"] = True
        goal.pop("point", None)
        goal.pop("body", None)
        goal.pop("aim", None)
        self._approach_arrived = False
        self._walk_released = False
        self._arm_wait0 = None
        self._gaze_locked = False
        self._need_gaze_plan = True
        self._jpeg_was_gaze = False
        self._pi0_approach_hold = False
        if self._grasp_tries >= GRASP_TRIES:
            goal["done"] = True
            self._note("Хват не получился.")
        else:
            self._note(note or "Хват мимо. Смотрю ещё раз.")

    def _grasp_progress(self, goal: dict, frame: dict) -> bool:
        """Track contact. Success is world motion or L1 verify — not a timed poke on a fixed body."""
        if self._grasp_mark == int(self._tick):
            return bool(goal.get("stale"))
        self._grasp_mark = int(self._tick)
        if not self._approach_arrived or goal.get("done") or goal.get("target") == "head":
            return False
        # Neural/π0 near-field: no geometric miss clock while π0 owns the arm.
        if "u" in goal and self._pi0_driving and not goal.get("held"):
            self._grasp_tick0 = None
            return False
        if "u" in goal and not self._pi0_driving and not goal.get("held"):
            near = bool(float(getattr(self, "_reach_gap", 1.0)) <= TOUCH_M * 2.0)
            if not near:
                self._grasp_tick0 = None
                return False
        body = goal.get("body")
        if body is None or goal.get("point") is None:
            return False
        # Door panel contact with a hanging hand is not a grasp — palm must be at the mark.
        holding = hands_holding(self.model, self.data, str(goal.get("hand") or ""), int(body)) and palm_near_point(
            self.model, self.data, str(goal.get("hand") or ""), goal["point"], TOUCH_M * 2.5
        )
        dt = float(self.model.opt.timestep)
        if holding:
            self._grasp_tick0 = None
            self._slip_tick0 = None
            if not goal.get("held"):
                goal["held"] = True
                goal["z0"] = float(self.data.xpos[int(body)][2])
                goal["lift0"] = float(np.asarray(goal["point"])[2])
                goal["x0"] = np.asarray(self.data.xpos[int(body)], dtype=np.float64).copy()
                goal["world_moved"] = False
            if body_can_move(self.model, int(body)):
                if self._contact_tick0 is None:
                    self._contact_tick0 = int(self._tick)
                climbed = LIFT_MPS * (int(self._tick) - int(self._contact_tick0)) * dt
                z = float(self.data.xpos[int(body)][2])
                if z - float(goal["z0"]) >= LIFT_M:
                    goal["done"] = True
                    goal["world_moved"] = True
                    self._world_moved = True
                    self._note("Предмет в руке.")
                else:
                    pt = np.asarray(goal["point"], dtype=np.float64).copy()
                    pt[2] = float(goal["lift0"]) + min(climbed, LIFT_M + 0.02)
                    goal["point"] = pt
                    if abs(pt[2] - float(getattr(self, "_lift_sol_z", pt[2]))) > 0.01:
                        self._lift_sol_z = float(pt[2])
                        self._reach_tick = -10**9
            else:
                # Hinge door / button / static: contact alone is not success.
                if self._contact_tick0 is None:
                    self._contact_tick0 = int(self._tick)
                x0 = goal.get("x0")
                if x0 is not None:
                    delta = float(np.linalg.norm(np.asarray(self.data.xpos[int(body)], dtype=np.float64) - x0))
                    if delta >= 0.025:
                        goal["world_moved"] = True
                        self._world_moved = True
                        goal["done"] = True
                        self._note("Объект сдвинулся.")
                self._need_verify = True
            return False
        if goal.get("held"):
            if self._slip_tick0 is None:
                self._slip_tick0 = int(self._tick)
            if (int(self._tick) - int(self._slip_tick0)) * dt >= CONTACT_S:
                self._drop_grasp(goal, frame, "Хват сорвался. Смотрю ещё раз.")
                return True
            return False
        if self._pi0_driving or self._pi0_owns_near():
            return False
        if self._grasp_tick0 is None:
            self._grasp_tick0 = int(self._tick)
        if (int(self._tick) - int(self._grasp_tick0)) * dt >= GRASP_S:
            self._drop_grasp(goal, frame, "Хват мимо. Смотрю ещё раз.")
            return True
        return False

    def _shift_followed(self, vx: float, vy: float) -> bool:
        """True while the pelvis is still making progress along the clearance step."""
        yaw = self._heading()
        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        cmd = heading * float(vx) + left * float(vy)
        n = float(np.linalg.norm(cmd))
        if n < 0.05:
            return True
        key = (round(float(vx), 2), round(float(vy), 2))
        if getattr(self, "_shift_key", None) != key:
            self._shift_key = key
            self._shift_xy0 = None
            self._blocked_tick0 = int(self._tick)
        now = np.asarray(self._pelvis()[:2], dtype=np.float64)
        origin = getattr(self, "_shift_xy0", None)
        if origin is None:
            self._shift_xy0 = now.copy()
            return True
        moved = float(np.dot(now - origin, cmd / n))
        if moved > 0.04:
            self._shift_xy0 = now.copy()
            self._blocked_tick0 = int(self._tick)
            return True
        held = (int(self._tick) - int(self._blocked_tick0 or self._tick)) * float(self.model.opt.timestep)
        return held < 3.0

    def _stall(self, goal: dict, frame: dict, note: str) -> bool:
        if self._grasp_mark == int(self._tick):
            return bool(goal.get("stale"))
        if self._blocked_tick0 is None:
            self._blocked_tick0 = int(self._tick)
        held = (int(self._tick) - int(self._blocked_tick0)) * float(self.model.opt.timestep)
        if held < GRASP_S:
            return False
        self._grasp_mark = int(self._tick)
        self._drop_grasp(goal, frame, note)
        return True

    def _servo_reach(self, cmd: np.ndarray) -> np.ndarray:
        if self._awaiting_plan:
            return cmd
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = frame.get("reach") if frame else None
        if not goal or self.outcome == "fall":
            self._reach_sol = None
            if frame is not None:
                frame["approaching"] = False
            return cmd
        self._cast_look(goal)
        if goal.get("stale"):
            frame["approaching"] = False
            self._reach_sol = None
            return cmd
        if goal.get("unresolved") or ("u" in goal and goal.get("point") is None):
            self._touch_goal = True
            frame["approaching"] = False
            # No pixel yet. A student residual here walks him off while he is leaning.
            held = np.array(cmd, dtype=np.float32, copy=True)
            held[0] = np.float32(0.0)
            held[1] = np.float32(0.0)
            held[2] = np.float32(0.0)
            held[CMD_ARMS] = arm_hang_cmd()
            return held
        self._note_touch(goal)
        out = np.array(cmd, dtype=np.float32, copy=True)
        world = goal.get("point") is not None and goal.get("target") != "head"
        # The lean is only for the picture. Walk waits until the waist is back
        # and the stand has stopped rocking, or the first step falls backward.
        if world and not self._walk_released:
            waist = abs(float(self.data.qpos[self.qadr[WAIST_P]]))
            rocking = float(np.linalg.norm(self.data.qvel[0:6]))
            if waist > 0.05 or rocking > 0.25:
                out[0] = np.float32(0.0)
                out[1] = np.float32(0.0)
                out[2] = np.float32(0.0)
                out[CMD_ARMS] = arm_hang_cmd()
                frame["approaching"] = False
                self._reach_sol = None
                return out
            self._walk_released = True
        # Looking at the grasp point uses waist pitch — do not freeze the reach while leaned.
        if world and self._approach_arrived and self._reach_sol is not None and goal.get("point") is not None:
            # Solver gap is only refreshed every few ticks. The palm can already be
            # on the handle while the cache still says 15 cm — that blocked the pull.
            live = goal_distance(self.model, self.data, goal)
            if live is not None:
                self._reach_gap = min(float(getattr(self, "_reach_gap", live)), float(live))
        if world and self._approach_arrived:
            self._draw_contact(goal)
        # The grid is for finding a pose. Once he is there, a short jacobian
        # step keeps the palm on the point while the gait sways. A full grid
        # is ~100 ms and would hitch the stand.
        track = (
            world
            and self._approach_arrived
            and self._reach_sol is not None
            and float(getattr(self, "_reach_gap", 1.0)) <= TOUCH_M * 3.0
            and int(self._tick) - int(self._reach_tick) >= 25
        )
        if track:
            self._reach_sol, gap = refine_arm(
                self.model, self.data, goal, self.qadr, self.vadr, self._reach_sol
            )
            self._reach_gap = float(gap)
            self._reach_tick = int(self._tick)
        elif self._reach_sol is None or int(self._tick) - int(self._reach_tick) >= (
            100
            if world and not self._reach_close and float(getattr(self, "_reach_gap", 1.0)) < 0.30
            else 300
        ):
            self._reach_sol, gap = solve_arm(self.model, self.data, goal, self.qadr, self.vadr)
            self._reach_gap = float(gap)
            self._reach_tick = int(self._tick)
            shift = goal.get("shift")
            key = None if shift is None else (round(float(shift[0]), 2), round(float(shift[1]), 2))
            if key != getattr(self, "_shift_print", None):
                self._shift_print = key
                pelvis = self._pelvis()
                print(
                    f"reach gap={float(gap):.2f} shift={key} pelvis={float(pelvis[0]):.2f},{float(pelvis[1]):.2f} yaw={self._heading():.2f}",
                    flush=True,
                )
        was_approaching = bool(frame.get("approaching"))
        gap = float(getattr(self, "_reach_gap", 1.0))
        point = np.asarray(goal["point"], dtype=np.float64) if goal.get("point") is not None else None
        # The cached gap is the same on every physics tick. Counting it each tick
        # cancelled the step after 6 ms and the stand fell backward.
        just_solved = int(self._reach_tick) == int(self._tick)
        if world and point is not None and not self._reach_close and just_solved:
            best = float(goal.get("best_gap", gap + 1.0))
            here = np.asarray(self._pelvis()[:2], dtype=np.float64)
            origin = goal.get("flat_xy")
            moved = 0.0 if origin is None else float(np.linalg.norm(here - np.asarray(origin, dtype=np.float64)))
            if gap < best - 0.02:
                goal["best_gap"] = float(gap)
                goal["flat"] = 0
                goal["flat_xy"] = here.copy()
            elif moved > 0.10:
                goal["flat"] = int(goal.get("flat") or 0) + 1
                goal["flat_xy"] = here.copy()
                if gap < best:
                    goal["best_gap"] = float(gap)
            elif origin is None:
                goal["flat_xy"] = here.copy()
            # A new look from a few centimetres away hits a different object.
            # Stop only after the body has actually travelled and the hand is no closer.
            if int(goal.get("flat") or 0) >= 4 and gap > TOUCH_M:
                goal["shift"] = None
        # A clearance search can stop while the palm is still short. Creep toward
        # the point until a later solve can actually put the hand on it.
        if (
            world
            and point is not None
            and just_solved
            and gap > TOUCH_M * 1.5
            and goal.get("shift") is None
            and not self._reach_close
            and int(goal.get("flat") or 0) < 4
        ):
            delta = np.asarray(point[:2], dtype=np.float64) - np.asarray(self._pelvis()[:2], dtype=np.float64)
            dist = float(np.linalg.norm(delta))
            if dist > TOUCH_M:
                goal["shift"] = delta / dist * min(0.12, dist)
        if world and gap <= TOUCH_M * 1.5:
            self._reach_close = True
            goal["best_gap"] = float(gap)
            goal["shift"] = None
        if self._reach_close:
            goal["shift"] = None
        approaching = bool(
            world
            and point is not None
            and gap > RAISE_GAP_M
            and goal.get("shift") is not None
            and not self._reach_close
        )
        frame["approaching"] = approaching
        # Flat cleared the walk while the palm was still short (e.g. gap 0.39): raise
        # from here instead of freezing with hang arms forever.
        stranded = bool(
            world
            and point is not None
            and goal.get("shift") is None
            and int(goal.get("flat") or 0) >= 4
            and gap <= 0.55
            and not self._reach_close
        )
        raise_arm = bool(world and (self._reach_close or gap <= RAISE_GAP_M or stranded))
        if raise_arm:
            frame["approaching"] = False
            approaching = False
            goal["shift"] = None
        if world and not raise_arm:
            out[CMD_ARMS] = arm_hang_cmd()
        if approaching and not raise_arm:
            shift = np.asarray(goal["shift"], dtype=np.float64)
            yaw = self._heading()
            if float(np.linalg.norm(shift[:2])) > TOUCH_M:
                heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
                fwd = float(np.dot(np.asarray(shift[:2], dtype=np.float64), heading))
                if fwd < -TOUCH_M:
                    # Still yaw toward the stance point while backing — zero wz left him facing away.
                    dest = np.asarray(self._pelvis(), dtype=np.float64).copy()
                    dest[0] += float(shift[0])
                    dest[1] += float(shift[1])
                    _vx, wz = velocity_toward(yaw, self._pelvis(), dest)
                    vx = float(np.clip(fwd * 1.5, -0.25, 0.0))
                    vy = 0.0
                else:
                    dest = np.asarray(self._pelvis(), dtype=np.float64).copy()
                    dest[0] += float(shift[0])
                    dest[1] += float(shift[1])
                    vx, wz = velocity_toward(yaw, self._pelvis(), dest)
                    vy = 0.0
                    vx = float(np.clip(vx, -0.22, 0.22))
                # Near the point, yaw-only commands make the gait stomp without progress.
                if abs(vx) < 0.05 and gap <= RAISE_GAP_M + 0.08:
                    goal["shift"] = None
                    out[0] = np.float32(0.0)
                    out[1] = np.float32(0.0)
                    out[2] = np.float32(0.0)
                    frame["approaching"] = False
                    approaching = False
                    raise_arm = True
                else:
                    out[0] = np.float32(vx)
                    out[1] = np.float32(vy)
                    out[2] = np.float32(wz)
                    if self._shift_followed(vx, vy):
                        self._blocked_tick0 = None
                    elif self._stall(goal, frame, "Рука не проходит. Смотрю ещё раз."):
                        return out
            else:
                self._blocked_tick0 = None
            if approaching:
                speed = max(abs(float(out[0])), abs(float(out[1])), abs(float(out[2])))
                if speed > 0.05:
                    return out
                frame["approaching"] = False
                out[0] = np.float32(0.0)
                out[1] = np.float32(0.0)
                out[2] = np.float32(0.0)
                out[CMD_ARMS] = arm_hang_cmd()
                return out
        if world and not raise_arm:
            out[0] = np.float32(0.0)
            out[1] = np.float32(0.0)
            out[2] = np.float32(0.0)
            out[CMD_ARMS] = arm_hang_cmd()
            return out
        if world:
            if not self._approach_arrived:
                self._arm_wait0 = int(self._tick)
                # Keep the approach world point as memory. Close relook may refine
                # only if the handle is still visible — otherwise restore the anchor.
                if not getattr(self, "_relook_used", False):
                    self._relook_used = True
                    self._approach_arrived = True
                    self._save_anchor(goal)
                    # Close FOV often loses the handle onto the wall above the stove.
                    # Keep the approach world point — do not spam L1 from the same pose.
                    self._need_gaze_plan = False
                    self._gaze_locked = True
                    frame["approaching"] = False
                    self._note("Подошёл. Держу точку с подхода.")
                    out[0] = np.float32(0.0)
                    out[1] = np.float32(0.0)
                    out[2] = np.float32(0.0)
                    out[CMD_ARMS] = arm_hang_cmd()
                    return out
                vla = getattr(self, "vla", None)
                note = "На месте. Руки — π0." if getattr(vla, "_ready", False) else "На месте. Руки — IK."
                self._note(note)
                if vla is not None:
                    vla.reset_chunk()
                self._need_verify = True
                self._verify_at = 0.0
            self._approach_arrived = True
            out[0] = np.float32(0.0)
            out[1] = np.float32(0.0)
            out[2] = np.float32(0.0)
            if was_approaching:
                self._queue_tick0 = int(self._tick)
            # π0 owns near-field when a chunk is playing. Otherwise geometric IK.
            if self._pi0_driving:
                if self._grasp_progress(goal, frame):
                    return out
                return out
            if self._reach_sol is None or int(self._tick) - int(self._reach_tick) >= 25:
                self._reach_sol, gap = solve_arm(self.model, self.data, goal, self.qadr, self.vadr)
                self._reach_gap = float(gap)
                self._reach_tick = int(self._tick)
            if self._grasp_progress(goal, frame):
                return out
            return servo_reach(self.model, self.data, out, goal, self.qadr, self._reach_sol)
        return servo_reach(self.model, self.data, out, goal, self.qadr, self._reach_sol)

    def _lesson_block(self) -> str | None:
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        reach = (frame or {}).get("reach") or {}
        if "u" in reach and not reach.get("held"):
            return "Сон пропущен: рука не коснулась."
        return touch_lesson(bool(self._touch_goal), self._touch_d0, self._touch_best)

    def _load_queue(self, plan: Plan) -> None:
        attach_reach(plan, self.user_cmd or plan.instruction)
        attach_arm_channel(plan, self.user_cmd or plan.instruction)
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
        reach = frame.get("reach") or {}
        if "u" in reach and not reach.get("done"):
            return
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
        frame = self._queue[self._queue_i]
        cmd = np.array(frame["cmd"], dtype=np.float32, copy=True)
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
                finished = self._attempt_done()
                if self._ep_armed and not finished and not self._ep_slept and not self._ep_failed:
                    self._note("Команда прервана новой.")
                self._finish_episode(train=finished)
            self._awaiting_plan = True
            self._ep_armed = True
            self._ep_slept = False
            self._ep_failed = False
            self._quiet_tick = None
            self._touch_goal = False
            self._touch_d0 = None
            self._touch_best = None
            self._blocked_tick0 = None
            self._approach_arrived = False
            self._reach_close = False
            self._relook_used = False
            self._walk_released = False
            self._arm_wait0 = None
            self._gaze_locked = False
            self._need_gaze_plan = False
            self._jpeg_was_gaze = False
            self._marker_pos = None
            self._marker_aim = None
            self._marker_dim = False
            self._anchor = None
            self._pi0_prefetch = False
            self._pi0_approach_hold = False
            self._verify_at = 0.0
            self._need_verify = False
            self._verify_hold_noted = False
            self._world_moved = False
            self._wall_cast_tries = 0
            self._grasp_tick0 = None
            self._slip_tick0 = None
            self._contact_tick0 = None
            self._grasp_mark = -1
            self._grasp_tries = 0
            self._shift_xy0 = None
            self._shift_key = None
            self._ep_tick0 = int(self._tick)
            self._reset_pi0_episode()
            self._reset_e2e_episode()
            self.intent = text
            self.user_cmd = text
            if self._e2e_mode:
                self._e2e_prefetch = True
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
            if wants_flip(text):
                self._start_flip()

    def _start_flip(self) -> None:
        from agent.flip_policy import FlipPolicy

        pol = FlipPolicy(clip="flip_04.npz")
        here = (float(self.data.qpos[0]), float(self.data.qpos[1]))
        pol.reset(self.model, self.data, self.qadr, xy=here)
        self._flip = pol
        self._flip_q = pol.joint_pos[0].copy()
        self._flip_sub = 0
        self._awaiting_plan = False
        self.outcome = "ok"
        self.status = "сальто"
        if self.walk is not None:
            self.walk.reset()

    def _step_flip(self) -> None:
        from agent.flip_policy import flip_torque

        pol = self._flip
        nsub = max(1, int(round(0.02 / float(self.model.opt.timestep))))
        if self._flip_sub % nsub == 0 and not pol.done:
            self._flip_q = pol.act(self.model, self.data, self.qadr, self.vadr)
        self._flip_sub += 1
        self.data.ctrl[:N_ACT] = flip_torque(self.model, self.data, self._flip_q, pol, self.qadr, self.vadr)
        if self.finger_qadr.size:
            self.data.ctrl[N_ACT:] = 0
        mujoco.mj_step(self.model, self.data)
        self.q_cmd = self._flip_q
        self.ctrl_source = "flip"
        self._tick += 1
        if pol.done and self._flip_sub % nsub == 0:
            self._flip = None
            self.user_cmd = ""
            self.intent = ""
            self.status = "сальто готово"

    def apply_plan(self, plan: Plan, *, fresh: bool, l1_ok: bool = True):
        with self.lock:
            self.last_l1 = time.monotonic()
            self.l1_ok = bool(l1_ok)
            self._awaiting_plan = False
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
                old = self._queue[0]
                frame = self._frame(plan.queue[0])
                kept = old.get("reach") or {}
                new_reach = frame.get("reach") or {}
                if getattr(self, "_jpeg_was_gaze", False) and "u" in new_reach:
                    # Close relook may refine; junk casts fall back to the approach anchor.
                    merged = dict(kept)
                    merged["u"] = float(new_reach["u"])
                    merged["v"] = float(new_reach.get("v", kept.get("v", 0.5)))
                    if "hand" in new_reach:
                        merged["hand"] = new_reach["hand"]
                    merged.pop("point", None)
                    merged.pop("body", None)
                    merged.pop("aim", None)
                    merged.pop("drawn", None)
                    merged.pop("stale", None)
                    frame["reach"] = merged
                    self._reach_sol = None
                    self._reach_tick = -10**9
                    self._reach_close = False
                    self._grasp_tick0 = None
                    self._slip_tick0 = None
                    self._contact_tick0 = None
                elif getattr(self, "_jpeg_was_gaze", False) and "u" not in new_reach:
                    # Handle not visible up close — keep / restore the approach world point.
                    frame["reach"] = dict(kept)
                    if frame["reach"].get("point") is None:
                        self._restore_anchor(frame["reach"])
                    frame["approaching"] = False
                elif kept.get("held"):
                    frame["reach"] = kept
                    frame["approaching"] = bool(old.get("approaching"))
                elif kept.get("point") is not None and not kept.get("stale"):
                    # Freeze the world target while walking. Mid-walk L1 must not recast onto walls.
                    frame["reach"] = kept
                    frame["approaching"] = bool(old.get("approaching"))
                elif kept.get("stale") and not kept.get("done"):
                    new = frame.get("reach") or {}
                    if "u" not in new:
                        frame["reach"] = kept
                        frame["approaching"] = False
                    else:
                        self._approach_arrived = False
                        self._reach_close = False
                        self._arm_wait0 = None
                        self._pi0_approach_hold = False
                        self._reach_sol = None
                        self._reach_tick = -10**9
                        self._blocked_tick0 = None
                        self._grasp_tick0 = None
                        self._slip_tick0 = None
                        self._contact_tick0 = None
                had_yaw = old["yaw"] is not None
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
            frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
            reach = (frame or {}).get("reach") or {}
            if getattr(self, "_jpeg_was_gaze", False):
                self._need_gaze_plan = False
                self._jpeg_was_gaze = False
                if "u" in reach and reach.get("point") is None:
                    self._gaze_locked = True
                    self._note("Повторный кадр. Точка с него.")
                elif reach.get("point") is not None:
                    self._gaze_locked = True
                    self._note("Близко ручка не видна — цель с подхода.")
                else:
                    self._gaze_locked = False
                    self._note("Объект не виден — меняю ракурс.")
            elif "u" in reach and not self._gaze_locked:
                # First plan already pointed on the upright head JPEG — cast it, do not lean.
                self._gaze_locked = True
                self._need_gaze_plan = False
                self._note("Точка с прямого кадра.")
            if "u" in reach:
                import os

                if self._e2e_mode:
                    self._e2e_prefetch = True
                elif os.getenv("PI0_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}:
                    self._pi0_prefetch = True
            elif self._e2e_mode and fresh and (self.user_cmd or self.intent):
                # Whole-body e2e does not need a grasp UV to start.
                self._e2e_prefetch = True
            if fresh or jumped:
                self._clear_errors()

    def validate_grasp_uv(self, u: float, v: float) -> tuple[bool, int | None]:
        """Physics check for a vision pixel before the sim trusts it."""
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            return cast_grasp_uv(self.model, self.data, int(self.pelvis_id), float(u), float(v))

    def wants_gaze_point(self) -> bool:
        """Object miss: ask for a new pixel on the current upright frame, not a new gait."""
        with self.lock:
            return bool(getattr(self, "_need_gaze_plan", False) and getattr(self, "_jpeg_was_gaze", False))

    def wants_verify(self) -> bool:
        """Ask L1 whether the language goal succeeded — not a contact timer."""
        with self.lock:
            if self.l1_busy or not self.user_cmd or self._awaiting_plan:
                return False
            if getattr(self, "_need_gaze_plan", False):
                return False
            if not self._approach_arrived or not self._pi0_task():
                return False
            if not getattr(self, "_need_verify", False):
                return False
            frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
            reach = (frame or {}).get("reach") or {}
            if reach.get("done") or reach.get("stale"):
                return False
            wait0 = int(getattr(self, "_arm_wait0", None) or getattr(self, "_queue_tick0", self._tick))
            since = (int(self._tick) - wait0) * float(self.model.opt.timestep)
            if since < 4.0:
                return False
            # Ollama + GPU already fight the sim — never hammer verify every 2.5 s.
            now = time.monotonic()
            if now - float(getattr(self, "_verify_at", 0.0)) < 12.0:
                return False
            return True

    def apply_verify(self, result: dict) -> None:
        with self.lock:
            self._verify_at = time.monotonic()
            self._need_verify = False
            self.last_l1 = time.monotonic()
            frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
            reach = (frame or {}).get("reach") or {}
            done = bool(result.get("done"))
            retry = str(result.get("retry") or "hold").strip().lower()
            print(f"verify done={done} retry={retry}", flush=True)
            if done:
                reach["done"] = True
                self._note("L1: задача выполнена.")
                return
            if retry in {"repoint", "look", "point"}:
                if getattr(self, "_anchor", None) and self._approach_arrived:
                    # Keep IK on the approach point. Do not re-arm verify — that
                    # loop burned GPU/Ollama and froze the UI (сеть к якорю spam).
                    self._need_gaze_plan = False
                    self._need_verify = False
                    if not getattr(self, "_verify_hold_noted", False):
                        self._verify_hold_noted = True
                        self._note("L1: ещё не сделано — IK к якорю.")
                else:
                    self._need_gaze_plan = True
                    self._gaze_locked = False
                    self._jpeg_was_gaze = False
                    self._note("L1: ещё не сделано — уточняю точку.")
            else:
                # hold: allow one more check much later, not immediately
                self._need_verify = False
                self._note("L1: ещё не сделано — продолжаю.")

    def defer_l1(self, seconds: float) -> None:
        """Skip the next planner calls. Used when pointing already failed."""
        with self.lock:
            self.last_l1 = time.monotonic() + max(0.0, float(seconds)) - L1_PERIOD

    def l1_due(self) -> bool:
        with self.lock:
            if self.l1_busy or not self.user_cmd:
                return False
            if getattr(self, "_need_gaze_plan", False):
                return self._gaze_ready()
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
        step = np.clip(desired.astype(np.float32) - q, -SLEW, SLEW)
        if self._pi0_driving or self._e2e_driving or (self._approach_arrived and self._pi0_task() and not self._e2e_mode):
            # q_cmd is actuator order (29). qadr is qpos addresses and must not index it.
            # 1 rad/s. A chunk, or a solved pose, can ask for the whole arm at once.
            # Once the palm is latched — or nearly there — let the wrist finish turning.
            # The crawl left fingers hanging under the bone on the handle.
            if self._e2e_driving:
                arm_step = 0.006
            else:
                arm_step = 0.008 if (self._reach_close or float(getattr(self, "_reach_gap", 1.0)) <= RAISE_GAP_M) else 0.002
            step[15:29] = np.clip(desired.astype(np.float32)[15:29] - q[15:29], -arm_step, arm_step)
            if self._e2e_driving:
                step[0:15] = np.clip(desired.astype(np.float32)[0:15] - q[0:15], -arm_step, arm_step)
        if self._holding_gaze():
            # Relook needs an upright frame — ease waist back slowly.
            slow = desired.astype(np.float32)
            step[WAIST_P] = np.clip(slow[WAIST_P] - q[WAIST_P], -0.0015, 0.0015)
            step[WAIST_Y] = np.clip(slow[WAIST_Y] - q[WAIST_Y], -0.0015, 0.0015)
            step[WAIST_R] = np.clip(slow[WAIST_R] - q[WAIST_R], -0.0015, 0.0015)
        elif abs(float(desired[WAIST_P] - q[WAIST_P])) > 0.02 or abs(float(desired[WAIST_Y] - q[WAIST_Y])) > 0.02:
            # Gaze tracking: faster than relook settle, slower than arm slew.
            step[WAIST_P] = np.clip(desired[WAIST_P] - q[WAIST_P], -0.004, 0.004)
            step[WAIST_Y] = np.clip(desired[WAIST_Y] - q[WAIST_Y], -0.004, 0.004)
        return np.clip(q + step, self.lo, self.hi)

    def _pd_torque(self, q_cmd: np.ndarray) -> np.ndarray:
        return compute_torques(self.model, self.data, q_cmd, self.kp, self.kd, self.qadr, self.vadr)

    def _reset_pi0_episode(self) -> None:
        self._ep_pi0 = False
        self._pi0_driving = False
        self._pi0_touched = set()
        self._pi0_x0 = None
        self._pi0_m0 = None
        self._pi0_wait0 = None
        self._pi0_wait_noted = False
        vla = getattr(self, "vla", None)
        if vla is not None:
            vla.reset_chunk()

    def _reset_e2e_episode(self) -> None:
        self._ep_e2e = False
        self._e2e_driving = False
        self._e2e_q = None
        self._e2e_prefetch = False
        e2e = getattr(self, "e2e", None)
        if e2e is not None:
            e2e.reset_chunk()

    def _e2e_active(self) -> bool:
        return bool(
            getattr(self, "_e2e_mode", False)
            and (self.user_cmd or self.intent)
            and not self._awaiting_plan
            and self.outcome != "fall"
        )

    def _maybe_load_e2e(self) -> None:
        if not getattr(self, "_e2e_prefetch", False):
            return
        e2e = getattr(self, "e2e", None)
        if e2e is None:
            self._e2e_prefetch = False
            return
        if getattr(e2e, "_ready", False) or e2e.error:
            self._e2e_prefetch = False
            if e2e.error and not getattr(self, "_e2e_fail_noted", False):
                self._e2e_fail_noted = True
                self._note("E2E недоступна — каскад L3 (включи GPU / HF_TOKEN).")
            return
        note = e2e.load_now()
        if note:
            self._note(note)
        if e2e.error and not getattr(self, "_e2e_fail_noted", False):
            self._e2e_fail_noted = True
            self._note("E2E недоступна — каскад L3 (включи GPU / HF_TOKEN).")
        self._e2e_prefetch = False

    def _e2e_command(self, proprio: np.ndarray) -> np.ndarray | None:
        e2e = getattr(self, "e2e", None)
        if e2e is None or not self._e2e_active() or not getattr(e2e, "_ready", False):
            return None
        if self._tick % VISION_STRIDE == 0 and e2e.wants_still_frame():
            e2e.submit(self._render_vla(), self.user_cmd or self.intent, proprio)
        return e2e.command(proprio, self.lo[:N_ACT], self.hi[:N_ACT])

    def _cmd_from_body(self, q_body: np.ndarray) -> np.ndarray:
        """Shadow 18-D label for the student while e2e owns absolute joints."""
        cmd = stand_command()
        q = np.asarray(q_body, dtype=np.float32).reshape(N_ACT)
        cmd[CMD_ARMS] = q[15:29]
        return cmd

    def _pi0_wait_expired(self) -> bool:
        """True when weights are up but the first chunk never arrived in time."""
        t0 = getattr(self, "_pi0_wait0", None)
        if t0 is None:
            return False
        return (time.monotonic() - float(t0)) >= PI0_WAIT_S

    def _pi0_task(self) -> bool:
        if self._awaiting_plan or self.outcome == "fall" or not (self.user_cmd or self.intent):
            return False
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        if not frame:
            return False
        reach = frame.get("reach") or {}
        if reach.get("target") == "head":
            return False
        return "u" in reach

    def _hands_ready(self) -> bool:
        """The body has reached the stance and the legs have settled, so the arm may start."""
        if not getattr(self, "_approach_arrived", False):
            return False
        if getattr(self, "_need_gaze_plan", False):
            return False
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        if frame is None or frame.get("approaching"):
            return False
        reach = frame.get("reach") or {}
        if "u" in reach and reach.get("point") is None:
            return False
        walk = getattr(self, "walk", None)
        data = getattr(self, "data", None)
        settled = (walk is None or walk.holding) and (
            data is None or float(np.linalg.norm(data.qvel[0:6])) < 0.12
        )
        if settled:
            return True
        wait0 = getattr(self, "_arm_wait0", None)
        if wait0 is None or getattr(self, "model", None) is None:
            return False
        waited = (int(self._tick) - int(wait0)) * float(self.model.opt.timestep)
        return waited >= 1.5

    def _render_vla(self) -> np.ndarray:
        if self._vla_eye is None:
            self._vla_eye = mujoco.Renderer(self.model, VLA_H, VLA_W)
        self._vla_eye.update_scene(self.data, camera="head", scene_option=self._view_opt)
        return np.ascontiguousarray(self._vla_eye.render())

    def _pi0_owns_near(self) -> bool:
        """After the walk, π0 drives the arms when weights are up (until wait times out)."""
        vla = getattr(self, "vla", None)
        if not (
            self._approach_arrived
            and self._pi0_task()
            and vla is not None
            and getattr(vla, "_ready", False)
        ):
            return False
        if self._pi0_wait_expired():
            if not getattr(self, "_pi0_wait_noted", False):
                self._pi0_wait_noted = True
                self._note(f"π0 молчит {PI0_WAIT_S:.0f} с — кисть на IK.")
            return False
        return True

    def _neural_near(self) -> bool:
        """After approach, the wrist is owned by π0 when ready — else geometric IK."""
        return bool(self._approach_arrived and self._pi0_task())

    def _maybe_load_pi0(self) -> None:
        """Load π0 on the sim thread (required by this build's safetensors path)."""
        if not getattr(self, "_pi0_prefetch", False):
            return
        vla = getattr(self, "vla", None)
        if vla is None:
            self._pi0_prefetch = False
            return
        if getattr(vla, "_ready", False) or vla.error:
            self._pi0_prefetch = False
            if vla.error and not getattr(self, "_pi0_fallback_noted", False):
                self._pi0_fallback_noted = True
                self._note("π0 недоступна — near-field кисть на IK.")
            return
        note = vla.load_now()
        if note:
            self._note(note)
        if vla.error and not getattr(self, "_pi0_fallback_noted", False):
            self._pi0_fallback_noted = True
            self._note("π0 недоступна — near-field кисть на IK.")
        self._pi0_prefetch = False

    def _plant_hold_cmd(self, cmd: np.ndarray) -> np.ndarray:
        out = np.array(cmd, dtype=np.float32, copy=True)
        out[0] = np.float32(0.0)
        out[1] = np.float32(0.0)
        out[2] = np.float32(0.0)
        # Keep current arm targets while waiting for the first π0 chunk.
        out[CMD_ARMS] = np.asarray(self.q_cmd[15:29], dtype=np.float32)
        return out

    def _pi0_command(self, proprio: np.ndarray) -> np.ndarray | None:
        if not self._pi0_task() or not self._hands_ready():
            return None
        if not getattr(self.vla, "_ready", False):
            return None
        if self._tick % VISION_STRIDE == 0 and self.vla.wants_still_frame():
            self.vla.submit(self._render_vla(), self.user_cmd or self.intent, proprio)
        arm = np.asarray(proprio[15:29], dtype=np.float32)
        cmd = self.vla.command(arm, self.lo[15:29], self.hi[15:29])
        if cmd is None:
            return None
        # Legs already walked here. The chunk's base speed would start a second gait under the arm.
        cmd[CMD_VX] = np.float32(0.0)
        cmd[CMD_VY] = np.float32(0.0)
        cmd[CMD_WZ] = np.float32(0.0)
        return cmd

    def _mark_pi0_frame(self) -> None:
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        if not frame:
            return
        frame["approaching"] = False
        # hold_s must not invent success — L1 verify / world motion set done.

    def _note_pi0_contact(self) -> None:
        if not self._pi0_driving or not self._arm_geoms:
            return
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            other = g2 if g1 in self._arm_geoms else g1 if g2 in self._arm_geoms else -1
            if other < 0:
                continue
            body = int(self.model.geom_bodyid[other])
            if body <= 0 or body in self._robot_bodies:
                continue
            self._pi0_touched.add(body)

    def _pi0_moved(self) -> bool:
        return pose_moved(self._pi0_x0, self._pi0_m0, self.data.xpos, self.data.xmat, self._pi0_touched)

    def _finger_command(self) -> np.ndarray:
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = frame.get("reach") if frame else None
        # Flat open (zeros) sticks thumbs into the hips and blocks the raise.
        # Stay curled for travel and grasp; π0 does not drive Dex3.
        return finger_target(goal)

    def _finger_torque(self) -> np.ndarray:
        target = self._finger_command()
        self._finger_q += np.clip(target - self._finger_q, -0.012, 0.012)
        q = np.asarray(self.data.qpos[self.finger_qadr], dtype=np.float32)
        qd = np.asarray(self.data.qvel[self.finger_vadr], dtype=np.float32)
        tau = FINGER_KP * (self._finger_q - q) - FINGER_KD * qd
        tau = tau + np.asarray(self.data.qfrc_bias[self.finger_vadr], dtype=np.float32)
        return np.clip(tau, self.finger_tau_lo, self.finger_tau_hi).astype(np.float32)

    def _write_ctrl(self) -> None:
        self.data.ctrl[:N_ACT] = self._pd_torque(self.q_cmd)
        if self.finger_qadr.size:
            self.data.ctrl[N_ACT:] = self._finger_torque()

    def _update_steps(self) -> None:
        if abs(self._active_vx()) > 0.08:
            self._walk_ticks += 1
            dt = float(self.model.opt.timestep)
            self._step_count = int(self._walk_ticks * dt / STEP_PERIOD)
            self._steps_done = self._step_count
        if self._turn_heading0 is not None:
            self._yaw_applied = self._achieved_yaw()

    def step(self):
        if not TORCH_GATE.acquire(blocking=False):
            # π0 is in native torch. Physics stays put: stepping a frozen walk pose falls.
            return
        try:
            self._step_torch()
        finally:
            TORCH_GATE.release()

    def _step_torch(self):
        if getattr(self, "_flip", None) is not None:
            self._step_flip()
            return
        err = self._balance_err()
        self.errors.append(err.copy())
        proprio = self._hinges()
        z = self.waypoint.z()
        errors = np.stack(self.errors)
        plan_cmd = self._plan_command()
        e2e_q = self._e2e_command(proprio) if self._e2e_mode else None
        if e2e_q is not None:
            if not self._ep_e2e:
                self._ep_e2e = True
                self._queue_tick0 = int(self._tick)
            self._e2e_driving = True
            self._e2e_q = e2e_q
            self._pi0_driving = False
            teacher = self._cmd_from_body(e2e_q)
            self.ctrl_source = "e2e"
        elif self._e2e_mode and self._e2e_active() and getattr(self.e2e, "_ready", False):
            # Waiting for first whole-body chunk — hold stand, do not run arms-only π0.
            self._e2e_driving = False
            self._pi0_driving = False
            if self._tick % VISION_STRIDE == 0 and self.e2e.wants_still_frame():
                self.e2e.submit(self._render_vla(), self.user_cmd or self.intent, proprio)
            teacher = stand_command()
            self.ctrl_source = "e2e-wait"
        elif self._e2e_mode:
            # E2E enabled but weights down — geometric cascade without π0 GPU fight.
            self._e2e_driving = False
            self._pi0_driving = False
            teacher = self._servo_reach(plan_cmd)
            self.ctrl_source = "l3"
        else:
            pi0 = self._pi0_command(proprio)
            if pi0 is not None:
                if not self._ep_pi0:
                    self._ep_pi0 = True
                    self._queue_tick0 = int(self._tick)
                    self._pi0_x0 = np.array(self.data.xpos, dtype=np.float64, copy=True)
                    self._pi0_m0 = np.array(self.data.xmat, dtype=np.float64, copy=True)
                    self._pi0_touched = set()
                self._pi0_wait0 = None
                self._pi0_driving = True
                self._mark_pi0_frame()
                teacher = pi0
                self.ctrl_source = "s15"
            elif self._pi0_owns_near():
                self._pi0_driving = False
                if self._pi0_wait0 is None:
                    self._pi0_wait0 = time.monotonic()
                if self._tick % VISION_STRIDE == 0 and self.vla.wants_still_frame():
                    self.vla.submit(self._render_vla(), self.user_cmd or self.intent, proprio)
                teacher = self._servo_reach(plan_cmd)
                self.ctrl_source = "s15-wait"
            else:
                self._pi0_driving = False
                teacher = self._servo_reach(plan_cmd)
                self.ctrl_source = "l3"
        language = encode_instr(
            (self.user_cmd or self.intent)
            if (self._pi0_driving or self._e2e_driving)
            else self.waypoint.param_text()
        )
        chosen = teacher
        if self._tick % VISION_STRIDE == 0:
            self._eye_rgb = self._render_eye()
            self._eye_seq += 1
            self._rays = self._head_rays()
        student = self._last_student
        if self.outcome != "fall" and self._tick % VISION_STRIDE == 0:
            hist = self._hist[: max(self._hist_n, 0)]
            student = self._student(self._eye_rgb, proprio, language, z, errors, history=hist if self._hist_n else None)
            self._last_student = student
            mse = float(np.mean((student - teacher) ** 2))
            self.shadow_mse = mse
            self._maybe_update_authority(mse, fell=self._fell())
            src = "e2e" if self._e2e_driving else ("s15" if self._pi0_driving else "l3")
            row = {
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
                "source": src,
            }
            if self._e2e_q is not None:
                row["q_body"] = np.asarray(self._e2e_q, dtype=np.float32).tolist()
            self.logs.append(row)
            if len(self.logs) > 4_000:
                drop = len(self.logs) - 3_000
                self.logs = self.logs[drop:]
                self._ep_i = max(0, int(self._ep_i) - drop)
        if self.outcome != "fall" and not self._awaiting_plan and not self._e2e_driving:
            chosen = self._mix(teacher, student)
            if self.alpha > 1e-8:
                self.ctrl_source = f"l3+l2-{self.stage}"
        self._last_teacher = teacher
        if self._e2e_driving and self._e2e_q is not None:
            self._cmd = self._cmd_from_body(self._e2e_q)
        elif self._pi0_driving:
            self._cmd = chosen
            self._cmd[CMD_VX] = np.float32(0.0)
            self._cmd[CMD_VY] = np.float32(0.0)
            self._cmd[CMD_WZ] = np.float32(0.0)
        elif self._holding_gaze():
            self._cmd = self._servo_reach(chosen)
            self._cmd[CMD_VX] = np.float32(0.0)
            self._cmd[CMD_VY] = np.float32(0.0)
            self._cmd[CMD_WZ] = np.float32(0.0)
            self._cmd[CMD_ARMS] = arm_hang_cmd()
        else:
            self._cmd = self._servo_reach(chosen)
            if self._pi0_task() and not self._approach_arrived and not self._e2e_mode:
                self._cmd[CMD_ARMS] = arm_hang_cmd()
        if self._tick % VISION_STRIDE == 0:
            err_now = self.errors[-1] if self.errors else np.zeros(3, dtype=np.float32)
            self._push_hist(chosen, err_now, self._rays)
        if self.outcome != "fall" and self._tick % DECIMATION == 0 and not self._e2e_driving:
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
        elif self._e2e_driving:
            self._last_a *= np.float32(0.0)
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
        bal = balance_delta(
            body_xy(off, yaw),
            body_xy(d_off, yaw),
            height_01=height_01(float(self._cmd[CMD_H])),
            vx=vx,
        )
        if self._e2e_driving and self._e2e_q is not None:
            q_des = np.asarray(self._e2e_q, dtype=np.float32) + np.float32(E2E_BALANCE) * bal
        else:
            q_des = q_from_action(self._cmd, self._last_a, self._gait_phi) + bal
        q_des = np.clip(q_des, self.lo, self.hi)
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = frame.get("reach") if frame else None
        if self._holding_gaze():
            # Stay upright while waiting for a relook JPEG — never lean toward the floor.
            q_des[WAIST_P] = np.float32(0.0)
            q_des[WAIST_Y] = np.float32(0.0)
            q_des[WAIST_R] = np.float32(0.0)
            q_des[WAIST_P] = np.clip(q_des[WAIST_P], self.lo[WAIST_P], self.hi[WAIST_P])
        elif self._should_look_at_goal(goal):
            wy, wp = self._look_at_waist(np.asarray(goal["point"], dtype=np.float64))
            q_des[WAIST_Y] = np.float32(wy)
            q_des[WAIST_P] = np.float32(wp)
            q_des[WAIST_R] = np.float32(0.0)
        self.q_cmd = self._slew(q_des)
        self._write_ctrl()
        cmd_v = np.array([self._cmd[CMD_VX], self._cmd[CMD_VY], self._cmd[CMD_WZ]], dtype=np.float32)
        arm_q = np.asarray(self.data.qpos[self.qadr[15:29]], dtype=np.float64)
        arms_clear = float(np.max(np.abs(arm_q - arm_hang_cmd()))) > 0.45
        arms_out = arms_clear or float(np.max(np.abs(self._cmd[CMD_ARMS] - arm_hang_cmd()))) > 0.45
        planted = bool(getattr(self, "_approach_arrived", False)) or float(np.max(np.abs(cmd_v))) < 0.05
        dynamic = bool(arms_out and not planted and not self._e2e_driving)
        if self.walk is not None and not self._e2e_driving:
            leg = self.walk.leg_torque(
                self.data,
                self.qadr,
                self.vadr,
                cmd_v,
                self.kp,
                self.kd,
                float(self.model.opt.timestep),
                dynamic=dynamic,
            )
            if leg is not None:
                self.data.ctrl[:12] = leg
                self.ctrl_source = "g1-walk"
            else:
                self.ctrl_source = "g1-stand"
        mujoco.mj_step(self.model, self.data)
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = frame.get("reach") if frame else None
        if goal and self._pi0_driving:
            self._note_pi0_contact()
        if goal and (self._pi0_driving or self._approach_arrived) and not self._e2e_driving:
            self._grasp_progress(goal, frame)
        if self.walk is not None and not self._e2e_driving and (self._tick + 1) % DECIMATION == 0 and not self.walk.holding:
            self.walk.update(
                self.data,
                self.qadr,
                self.vadr,
                cmd_v,
                dynamic=dynamic,
            )
        elif self.walk is not None and (self.walk.holding or self._e2e_driving):
            if self._e2e_driving:
                self.walk.reset()
            elif self.walk.holding:
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

    def orbit_camera(self, dx: float = 0.0, dy: float = 0.0, zoom: float = 0.0) -> None:
        """Orbit the operator camera. dx/dy are pixels, zoom is wheel deltaY."""
        with self.lock:
            off = self._cam_offset
            if dx:
                yaw = -float(dx) * 0.006
                c, s = float(np.cos(yaw)), float(np.sin(yaw))
                x, y = float(off[0]), float(off[1])
                off[0] = c * x - s * y
                off[1] = s * x + c * y
            if dy:
                horiz = float(np.hypot(off[0], off[1]))
                elev = float(np.arctan2(off[2], max(horiz, 1e-6)))
                elev = float(np.clip(elev - float(dy) * 0.006, -1.15, 1.15))
                length = float(np.linalg.norm(off))
                new_h = length * float(np.cos(elev))
                if horiz < 1e-6:
                    off[0], off[1] = new_h, 0.0
                else:
                    scale = new_h / horiz
                    off[0] *= scale
                    off[1] *= scale
                off[2] = length * float(np.sin(elev))
            if zoom:
                length = float(np.linalg.norm(off))
                length = float(np.clip(length * float(np.exp(float(zoom) * 0.0012)), 0.7, 9.0))
                norm = float(np.linalg.norm(off))
                if norm > 1e-6:
                    off *= length / norm
            self._kick_render = True

    def reset_demo_camera(self) -> None:
        with self.lock:
            self._cam_offset = np.array([-1.15, -1.70, 0.767], dtype=np.float64)
            self._kick_render = True

    def _place_demo_camera(self) -> None:
        origin = np.asarray(self._pelvis(), dtype=np.float64)
        off = np.asarray(self._cam_offset, dtype=np.float64)
        dist = float(np.linalg.norm(off))
        # Start past the torso. A ray from the pelvis hits the body itself.
        margin = 0.45
        if dist > margin + 0.2:
            vec = off / dist
            geomid = np.zeros(1, dtype=np.int32)
            groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            hit = mujoco.mj_ray(
                self.model,
                self.data,
                origin + vec * margin,
                vec,
                groups,
                1,
                -1,
                geomid,
            )
            if hit >= 0.0 and margin + float(hit) < dist:
                dist = max(margin + 0.15, (margin + float(hit)) * 0.9)
                off = vec * dist
        cid = int(self.model.camera("demo").id)
        self.model.cam_pos[cid] = origin + off
        mujoco.mj_camlight(self.model, self.data)

    def _draw_goal_markers(self, scn) -> None:
        """Paint the cast reach point into the operator preview. Survives brief stale gaps."""
        pos = getattr(self, "_marker_pos", None)
        aim = getattr(self, "_marker_aim", None)
        frame = self._queue[self._queue_i] if getattr(self, "_queue", None) else None
        goal = (frame or {}).get("reach") or {}
        if goal.get("point") is not None and not goal.get("done"):
            pos = np.asarray(goal["point"], dtype=np.float64)
            aim = goal.get("aim")
            self._marker_pos = pos.copy()
            if aim is not None:
                self._marker_aim = np.asarray(aim, dtype=np.float64).copy()
            self._marker_dim = False
        if pos is None or goal.get("done"):
            return
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        dim = bool(getattr(self, "_marker_dim", False) or goal.get("stale"))
        alpha = 0.45 if dim else 0.95
        if int(scn.ngeom) >= int(scn.maxgeom):
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.04, 0.0, 0.0], dtype=np.float64),
            pos=pos,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array([1.0, 0.2, 0.05, alpha], dtype=np.float32),
        )
        scn.ngeom += 1
        if aim is None or int(scn.ngeom) >= int(scn.maxgeom):
            return
        aim = np.asarray(aim, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(aim))
        if n < 1e-6:
            return
        aim = aim / n
        surface = pos + aim * max(0.0, float(STANDOFF_M))
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0.0, 0.0], dtype=np.float64),
            pos=surface,
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array([0.15, 0.85, 0.25, alpha * 0.95], dtype=np.float32),
        )
        scn.ngeom += 1
        if int(scn.ngeom) >= int(scn.maxgeom):
            return
        try:
            idx = int(scn.ngeom)
            mujoco.mjv_initGeom(
                scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3, dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array([1.0, 0.85, 0.1, alpha], dtype=np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[idx],
                mujoco.mjtGeom.mjGEOM_LINE,
                2.0,
                surface.reshape(3, 1),
                pos.reshape(3, 1),
            )
            scn.ngeom += 1
        except Exception:
            pass

    def _render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, PREVIEW_H, PREVIEW_W)
        self._place_demo_camera()
        self.renderer.update_scene(self.data, camera="demo", scene_option=self._view_opt)
        self._draw_goal_markers(self.renderer.scene)
        # JPEG runs on the stream thread. Copy now: the renderer reuses this buffer.
        self._preview_rgb = np.array(self.renderer.render(), copy=True)
        self._preview_seq += 1
        # Head look only on the sim thread — concurrent GL on Windows tears / flips.
        if self.look is None:
            self.look = mujoco.Renderer(self.model, LOOK_H, LOOK_W)
        self.look.update_scene(self.data, camera="head", scene_option=self._view_opt)
        self._look_rgb = np.array(self.look.render(), copy=True)
        self._look_seq += 1

    def jpeg(self) -> bytes:
        with self.lock:
            rgb = self._preview_rgb
            seq = self._preview_seq
            if seq == self._jpeg_seq:
                return self._jpeg
        if rgb is None:
            return b""
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        with self.lock:
            if seq >= self._jpeg_seq:
                self._jpeg = data
                self._jpeg_seq = seq
        return data

    def _render_eye(self) -> np.ndarray:
        if self.eye is None:
            self.eye = mujoco.Renderer(self.model, VISION_H, VISION_W)
        self.eye.update_scene(self.data, camera="head", scene_option=self._view_opt)
        return np.array(self.eye.render(), copy=True)

    def look_jpeg(self) -> bytes:
        """Operator head preview from the sim-thread look buffer (no GL here)."""
        with self.lock:
            rgb = self._look_rgb
            seq = self._look_seq
            if seq == self._look_jpeg_seq and self._look_jpeg:
                return self._look_jpeg
        if rgb is None:
            return b""
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
        data = buf.getvalue()
        with self.lock:
            if seq >= self._look_jpeg_seq:
                self._look_jpeg = data
                self._look_jpeg_seq = seq
        return data

    def planner_jpeg(self) -> bytes:
        """Head camera for L1 — same sim-thread buffer as the operator preview."""
        data = self.look_jpeg()
        self._jpeg_was_gaze = bool(getattr(self, "_need_gaze_plan", False) and self._gaze_ready())
        return data

    def eye_jpeg(self) -> bytes:
        with self.lock:
            rgb = self._eye_rgb
            seq = self._eye_seq
            if seq == self._eye_jpeg_seq and self._eye_jpeg:
                return self._eye_jpeg
            if rgb is None or int(np.asarray(rgb).sum()) == 0:
                return self._eye_jpeg
            rgb = np.array(rgb, copy=True)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=78)
        data = buf.getvalue()
        with self.lock:
            if seq >= self._eye_jpeg_seq:
                self._eye_jpeg = data
                self._eye_jpeg_seq = seq
        return data

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
                "s15": self.vla.phase(),
                "s15_s": None if self.vla.infer_s() is None else round(float(self.vla.infer_s()), 3),
                "s15_err": self.vla.error,
                "e2e": None if self.e2e is None else self.e2e.phase(),
                "e2e_s": None if self.e2e is None or self.e2e.infer_s() is None else round(float(self.e2e.infer_s()), 3),
                "e2e_err": "" if self.e2e is None else self.e2e.error,
                "e2e_on": bool(self._e2e_mode),
                "l1_ok": self.l1_ok,
                "l1_model": self.planner.model,
                "l1_url": self.planner.base_url,
                "l1_err": self.planner.last_err,
                "status": self.status,
                "skill": self._queue[self._queue_i]["skill"] if getattr(self, "_queue", None) else self.waypoint.skill,
                "params": self.waypoint.params,
                "queue_i": int(getattr(self, "_queue_i", 0)),
                "queue_len": len(getattr(self, "_queue", []) or []),
                "trials": [t.as_public(i + 1) for i, t in enumerate(self.trials.items())],
            }

    def _timer_period(self, ms: int) -> None:
        """Windows sleeps round 2 ms up to ~15 ms unless the timer is 1 ms."""
        if os.name != "nt":
            return
        if ms:
            ctypes.windll.winmm.timeBeginPeriod(ms)
            self._timer_ms = ms
        elif getattr(self, "_timer_ms", 0):
            ctypes.windll.winmm.timeEndPeriod(self._timer_ms)
            self._timer_ms = 0

    def loop(self, stop: threading.Event):
        self._timer_period(1)
        self._render()
        try:
            self._loop(stop)
        finally:
            self._timer_period(0)

    def _loop(self, stop: threading.Event):
        while not stop.is_set():
            job = None
            started = time.perf_counter()
            with self.lock:
                if not self._trial_busy:
                    if self._e2e_mode:
                        self._maybe_load_e2e()
                    else:
                        self._maybe_load_pi0()
                    self.step()
                    now = time.perf_counter()
                    if self._kick_render or now >= self._preview_at:
                        self._kick_render = False
                        self._preview_at = now + PREVIEW_INTERVAL
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
            remain = float(self.model.opt.timestep) - (time.perf_counter() - started)
            # Event.wait(1 ms) sleeps ~15 ms on Windows. Spin the short remainder.
            if remain > 0.0002:
                deadline = time.perf_counter() + remain
                while time.perf_counter() < deadline and not stop.is_set():
                    pass

    def close(self):
        for attr in ("renderer", "eye", "look"):
            r = getattr(self, attr)
            if r is None:
                continue
            try:
                r.close()
            except Exception:
                pass
            setattr(self, attr, None)
