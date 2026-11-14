"""Cognitive Robot Engine: System 0 (Balance/Walk) + System 1 (CMS VLA) + System 2 (Gemini/Ollama VLM) + SLEEP."""

from __future__ import annotations

import collections
import hashlib
import io
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from agent.config import (
    CHECKPOINT_DIR,
    COGNITIVE_DT,
    LOOK_H,
    LOOK_W,
    MICRO_SLEEP_IDLE_S,
    MICRO_SLEEP_MIN_STEPS,
    PHYSICS_DT,
    PREVIEW_H,
    PREVIEW_INTERVAL,
    PREVIEW_W,
    SLEEP_ACTION_DIM,
    SLEEP_HORIZON,
    SLEEP_IMG_SIZE,
    SLEEP_PROPRIO_DIM,
    STAND_Z,
    SUBSTEPS_PER_COG_STEP,
    TAU_THINK_DEFAULT,
)
from agent.flip_policy import wants_flip
from agent.h2 import MODEL_XML, N_ACT, actuator_addrs, disable_wrist_mesh_contacts, joint_limits
from agent.pi0_e2e import Pi0E2EModel, pi0_enabled
from agent.planner import VisionLanguagePlanner
from agent.sleep import FoundationVLAWithCMS, Learner, SleepConfig, tokenize
from agent.system0 import (
    System0Controller,
    default_raw_action,
    extract_proprio,
    norm_to_raw_action,
)
from agent.system2 import System2Executor


def resolve_device() -> torch.device:
    req = os.getenv("SLEEP_DEVICE", "").strip().lower()
    if req in ("cpu", "cuda"):
        return torch.device(req if req == "cpu" or torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CognitiveRobotEngine:
    """Unified runtime integrating MuJoCo G1 physics, Gemini/Ollama VLM, and SLEEP Wake/Sleep learning."""

    def __init__(self, *, headless: bool = False, defer_render: bool = False, enable_pi0: Optional[bool] = None) -> None:
        self.device = resolve_device()
        self.ckpt_path = CHECKPOINT_DIR / "sleep_vla.pt"
        self.headless = headless
        self._enable_pi0 = pi0_enabled() if enable_pi0 is None else bool(enable_pi0)

        # Physics setup
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        disable_wrist_mesh_contacts(self.model)
        qadr, vadr = actuator_addrs(self.model)
        lo, hi = joint_limits(self.model)
        self.qadr, self.vadr = qadr[:N_ACT], vadr[:N_ACT]
        self.lo, self.hi = lo[:N_ACT], hi[:N_ACT]
        self.sys0 = System0Controller(self.model, self.data, self.qadr, self.vadr, self.lo, self.hi)

        # Locks: _sim_lock for short physics/render ticks; _net_lock for PyTorch SLEEP weights
        self._sim_lock = threading.RLock()
        self._net_lock = threading.RLock()

        # Logs
        self._log_lock = threading.Lock()
        self._logs: collections.deque[Dict[str, str]] = collections.deque(maxlen=200)

        # SLEEP Policy (System 1) + Continual Learner
        self.vla = FoundationVLAWithCMS(
            act_dim=SLEEP_ACTION_DIM,
            state_dim=SLEEP_PROPRIO_DIM,
            horizon=SLEEP_HORIZON,
            d=256,
            chunks=(8, 24, 72),
            n_slots=(8, 8, 24),
            rank=16,
        ).to(self.device)
        self.learner = Learner(
            self.vla,
            n_tasks=32,
            cfg=SleepConfig(),
            use_sleep=True,
            log=lambda msg: self.log_event("sleep", msg),
        )
        loaded = self.learner.load_checkpoint(self.ckpt_path)

        # System 2 Cognitive Planner + E2E π₀ Backbone & Coordinator
        self.planner = VisionLanguagePlanner(log=lambda msg: self.log_event("warn" if "fallback" in msg.lower() else "info", msg))
        self.pi0 = Pi0E2EModel(
            self.device,
            self._net_lock,
            log=lambda msg: self.log_event("info" if "Готова" in msg else "warn", msg),
        )
        if self._enable_pi0:
            self.pi0.start_async()
        self.sys2 = System2Executor(
            self.model,
            self.data,
            self.sys0,
            self.planner,
            render_head_fn=self._get_head_rgb_blocking,
            step_cog_fn=self._step_cog_from_worker,
            is_aborted_fn=lambda: (not self.power_on) or self.standby_mode or self._abort_flag,
            log_fn=self.log_event,
            sim_lock=self._sim_lock,
            sys1_propose_fn=self._propose_sys1,
            pi0_propose_fn=self.pi0.propose_chunk,
            cms_slots_fn=self.vla.cms.used_slots,
            on_sys1_step_fn=self._tick_sys1_counter,
            tau_fn=lambda: self.tau_think,
        )

        # Hardware / Cognitive FSM state
        self.power_on = True
        self.standby_mode = False
        self.state = "IDLE"          # OFF | IDLE | SYS1 | SYS2 | MICRO_SLEEP | STANDBY
        self.tau_think = float(TAU_THINK_DEFAULT)
        self.current_uncertainty = 0.85
        self.latency_ms = 0.0
        self.current_command = "Ожидание команды..."
        self.sys1_count = 0
        self.sys2_count = 0
        self.total_sleep_cycles = 0

        # Active actuator target tracked by 500 Hz physics loop
        self._active_raw = default_raw_action()
        self._tuck_left = True
        self._tuck_right = True
        self._abort_flag = False
        self._executing = False
        self._last_activity = time.time()

        # Wake experience buffer: (rgb_128, proprio_28, command, chunk_H_A, task_id)
        self.wake_buffer: collections.deque[Tuple[np.ndarray, np.ndarray, str, np.ndarray, int]] = collections.deque(maxlen=600)
        self.failure_buffer: collections.deque[Tuple[np.ndarray, np.ndarray, str, np.ndarray, int]] = collections.deque(maxlen=200)
        self.unconsolidated_steps = 0
        self._unlearned_failures = 0
        self._sleep_done_for_batch = False
        self._task_vocab: dict[str, int] = {}
        self._pending_skills: set[str] = set()

        # Command queue for worker thread
        self._cmd_cv = threading.Condition()
        self._pending_cmd: Optional[str] = None
        self._active_goal: Optional[str] = None
        self._goal_retries_left: int = 0
        self._max_post_sleep_retries: int = 2
        self._worker_stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Rendering & camera state (owned strictly by sim thread when running live)
        self._preview_renderer: Optional[mujoco.Renderer] = None
        self._head_renderer: Optional[mujoco.Renderer] = None
        self._loop_thread_id: Optional[int] = None
        self._cam_offset = np.array([-0.32, -1.68, 0.68], dtype=np.float64)
        self._view_opt = mujoco.MjvOption()
        self._view_opt.geomgroup[:] = (0, 1, 1, 0, 0, 0)
        self._preview_rgb: Optional[np.ndarray] = None
        self._head_rgb: np.ndarray = np.zeros((LOOK_H, LOOK_W, 3), dtype=np.uint8)
        self._preview_seq = 0
        self._head_seq = 0
        self._jpeg_cache = b""
        self._jpeg_seq = -1
        self._head_jpeg_cache = b""
        self._head_jpeg_seq = -1
        self._last_render_ts = 0.0
        self._step_event = threading.Event()
        self._step_cv = threading.Condition()
        self._step_seq = 0

        with self._sim_lock:
            self.sys0.reset_pose(settle_ticks=120)
            if not defer_render:
                self._render_sync()

        # Pre-warm GPU kernels at startup so first inference doesn't show ~2s JIT lag
        if self.device.type == "cuda":
            try:
                dummy_img = torch.zeros(1, 3, SLEEP_IMG_SIZE, SLEEP_IMG_SIZE, device=self.device)
                dummy_st = torch.zeros(1, SLEEP_PROPRIO_DIM, device=self.device)
                dummy_tok = torch.zeros(1, 24, dtype=torch.long, device=self.device)
                with self._net_lock, torch.no_grad():
                    self.vla.predict_action(self.vla.encode(dummy_img, dummy_st, dummy_tok), greedy=True)
            except Exception:
                pass

        if loaded:
            self.log_event("sleep", f"Загружена память CMS: слотов {self.vla.cms.used_slots()}/{self.vla.cms.total_slots()}")
        standup_tag = "OK" if self.sys0.standup is not None else "none"
        self.log_event("info", f"Движок готов (device={self.device}, VLM={self.planner.active_backend}, standup={standup_tag}).")

    # ------------------------------------------------------------------ logging
    def log_event(self, level: str, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._log_lock:
            self._logs.append({"time": ts, "level": level, "text": text})
        print(f"[{ts}] [{level}] {text}", flush=True)

    def pop_logs(self) -> List[Dict[str, str]]:
        with self._log_lock:
            out = list(self._logs)
            self._logs.clear()
            return out

    @staticmethod
    def _skill_key(cmd: str) -> str:
        return " ".join(re.findall(r"[a-zа-яё0-9]+", cmd.lower()))

    def _effective_sigma(self, command: str, raw_sigma: float) -> float:
        key = self._skill_key(command)
        if key not in self.learner.consolidated_skills:
            return max(float(raw_sigma), 0.85)
        return float(raw_sigma)

    def _task_id(self, cmd: str) -> int:
        key = self._skill_key(cmd) or cmd.lower().strip()
        if key not in self._task_vocab:
            idx = len(self._task_vocab) % 32
            self._task_vocab[key] = idx
        return self._task_vocab[key]

    # ------------------------------------------------------------------ HAL / UI controls
    def _clear_goal_locked(self, *, clear_pending: bool = True) -> None:
        self._active_goal = None
        self._goal_retries_left = 0
        if clear_pending:
            self._pending_cmd = None

    def set_power(self, on: bool) -> None:
        self.power_on = bool(on)
        if not self.power_on:
            self._abort_flag = True
            self._executing = False
            with self._cmd_cv:
                self._clear_goal_locked()
            self.state = "OFF"
            self.log_event("warn", "Питание моторов отключено (E-Stop).")
        else:
            self._abort_flag = False
            self.state = "STANDBY" if self.standby_mode else "IDLE"
            self.log_event("info", "Питание включено. System 0 удерживает стойку.")

    def set_standby(self, on: bool, *, resume_goal: bool = True) -> None:
        self.standby_mode = bool(on)
        resumed_goal: Optional[str] = None
        if self.standby_mode:
            self._abort_flag = True
            self._executing = False
            self._active_raw = default_raw_action()
            self._tuck_left = True
            self._tuck_right = True
            self._sleep_done_for_batch = False
            self.state = "STANDBY"
            self.log_event("sleep", "Режим STAND-BY: стойка зафиксирована, запуск NREM/REM консолидации...")
        else:
            self._abort_flag = False
            self.state = "IDLE" if self.power_on else "OFF"
            self.log_event("info", "Выход из Stand-by: ассистент готов к командам.")
            with self._cmd_cv:
                if resume_goal and self.power_on and self._active_goal and self._pending_cmd is None:
                    self._pending_cmd = self._active_goal
                    self.current_command = self._active_goal
                    resumed_goal = self._active_goal
        with self._cmd_cv:
            self._cmd_cv.notify_all()
        if resumed_goal:
            self.log_event("info", f"[Wake-up] После сна продолжаю незавершённую задачу «{resumed_goal}»...")

    def set_tau_think(self, tau: float) -> None:
        self.tau_think = float(np.clip(tau, 0.05, 1.0))
        self.log_event("info", f"Порог уверенности τ_think = {self.tau_think:.2f}")

    def reset_sim(self) -> None:
        self._abort_flag = True
        self._executing = False
        with self._cmd_cv:
            self._clear_goal_locked()
        self.sys2.reset_memory()
        with self._sim_lock:
            self._active_raw = default_raw_action()
            self._tuck_left = True
            self._tuck_right = True
            self.sys0.reset_pose(settle_ticks=120)
            self._last_render_ts = 0.0
            if self._loop_thread_id is None or self._loop_thread_id == threading.get_ident():
                self._render_sync()
        self._abort_flag = False
        self.current_command = "Ожидание команды..."
        self.state = "STANDBY" if self.standby_mode else ("IDLE" if self.power_on else "OFF")
        self.log_event("info", "Позиция робота сброшена (Spawn).")

    def dispatch_command(self, text: str) -> Dict[str, Any]:
        raw = text.strip()
        if not raw:
            return {"ok": False, "error": "Пустая команда"}
        low = raw.lower()

        if any(w in low for w in ("стой", "стоп", "остановись", "stop", "halt")):
            self._abort_flag = True
            self._executing = False
            with self._cmd_cv:
                self._clear_goal_locked()
            self._active_raw = default_raw_action()
            self._tuck_left = True
            self._tuck_right = True
            self.state = "IDLE" if self.power_on else "OFF"
            self.log_event("warn", "Команда СТОП: движение прервано.")
            return {"ok": True, "action": "stop"}

        if any(w in low for w in ("спать", "уснуть", "в сон", "standby", "sleep")):
            self.set_standby(True)
            return {"ok": True, "action": "standby"}

        if any(w in low for w in ("проснуться", "подъем", "подъём", "wake")):
            self.set_standby(False)
            return {"ok": True, "action": "wake"}

        if any(w in low for w in ("сброс", "spawn", "reset", "заново")):
            self.reset_sim()
            return {"ok": True, "action": "reset"}

        if any(w in low for w in ("встать", "встань", "поднимись с пола", "standup", "stand up")):
            with self._sim_lock:
                started = self.sys0.start_standup()
            if started:
                self.state = "RECOVERING"
                self.log_event("info", "[System 0] Запуск политики вставания после падения (SuperMimic RL)...")
                return {"ok": True, "action": "standup"}
            return {"ok": False, "error": "Политика вставания не загружена"}

        if not self.power_on:
            self.log_event("warn", "Включите POWER ON для выполнения команды.")
            return {"ok": False, "error": "Power is OFF"}

        if self.standby_mode:
            self.set_standby(False, resume_goal=False)

        with self._cmd_cv:
            self._abort_flag = True
            self._pending_cmd = raw
            if wants_flip(raw):
                self._clear_goal_locked(clear_pending=False)
            else:
                self._active_goal = raw
                self._goal_retries_left = self._max_post_sleep_retries
            self._cmd_cv.notify_all()
        self.current_command = raw
        self._last_activity = time.time()
        return {"ok": True, "action": "queued", "command": raw}

    # ------------------------------------------------------------------ Camera & MJPEG
    def orbit_camera(self, dx: float = 0.0, dy: float = 0.0, zoom: float = 0.0) -> None:
        with self._sim_lock:
            yaw = -dx * 0.006
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            x, y, z = self._cam_offset
            nx, ny = c * x - s * y, s * x + c * y
            nz = float(np.clip(z + dy * 0.006, -0.35, 1.85))
            off = np.array([nx, ny, nz], dtype=np.float64)
            if zoom != 0.0:
                scale = float(np.clip(1.0 + zoom * 0.12, 0.5, 2.5))
                off *= scale
            dist = float(np.linalg.norm(off))
            if dist < 0.7:
                off = off / max(dist, 1e-6) * 0.7
            elif dist > 5.0:
                off = off / dist * 5.0
            self._cam_offset = off

    def reset_camera(self) -> None:
        with self._sim_lock:
            self._cam_offset = np.array([-0.32, -1.68, 0.68], dtype=np.float64)

    def _place_demo_camera(self) -> None:
        origin = np.asarray(self.data.xpos[self.sys0.pelvis_id], dtype=np.float64)
        cam_w = origin + self._cam_offset
        # Keep camera inside kitchen walls: x in [-1.38, 1.45], y in [-2.28, 3.00], z in [0.25, 2.75]
        cam_w[0] = float(np.clip(cam_w[0], -1.38, 1.45))
        cam_w[1] = float(np.clip(cam_w[1], -2.28, 3.00))
        cam_w[2] = float(np.clip(cam_w[2], 0.25, 2.75))
        cid = int(self.model.camera("demo").id)
        self.model.cam_pos[cid] = cam_w
        mujoco.mj_camlight(self.model, self.data)

    def _draw_markers(self, scn) -> None:
        pt = self.sys2.marker_pos
        if pt is None or int(scn.ngeom) >= int(scn.maxgeom):
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.035, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(pt, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array([0.1, 0.95, 0.45, 0.92], dtype=np.float32),
        )
        scn.ngeom += 1

    def _render_sync(self) -> None:
        if self.headless:
            return
        if self._preview_renderer is None:
            self._preview_renderer = mujoco.Renderer(self.model, PREVIEW_H, PREVIEW_W)
        if self._head_renderer is None:
            self._head_renderer = mujoco.Renderer(self.model, LOOK_H, LOOK_W)

        self._place_demo_camera()
        self._preview_renderer.update_scene(self.data, camera="demo", scene_option=self._view_opt)
        self._draw_markers(self._preview_renderer.scene)
        self._preview_rgb = np.array(self._preview_renderer.render(), copy=True)
        self._preview_seq += 1

        self._head_renderer.update_scene(self.data, camera="head", scene_option=self._view_opt)
        self._head_rgb = np.array(self._head_renderer.render(), copy=True)
        self._head_seq += 1

    def get_overview_jpeg(self) -> bytes:
        with self._sim_lock:
            rgb = self._preview_rgb
            seq = self._preview_seq
            if seq == self._jpeg_seq and self._jpeg_cache:
                return self._jpeg_cache
        if rgb is None:
            return b""
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=84)
        data = buf.getvalue()
        with self._sim_lock:
            if seq >= self._jpeg_seq:
                self._jpeg_cache = data
                self._jpeg_seq = seq
        return data

    def get_head_jpeg(self) -> bytes:
        with self._sim_lock:
            rgb = self._head_rgb
            seq = self._head_seq
            if seq == self._head_jpeg_seq and self._head_jpeg_cache:
                return self._head_jpeg_cache
        if rgb is None or int(rgb.sum()) == 0:
            return b""
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
        data = buf.getvalue()
        with self._sim_lock:
            if seq >= self._head_jpeg_seq:
                self._head_jpeg_cache = data
                self._head_jpeg_seq = seq
        return data

    def _get_head_rgb_blocking(self) -> np.ndarray:
        with self._sim_lock:
            return self._head_rgb.copy()

    def _step_cog_from_worker(self, raw: np.ndarray, tuck_left: bool = True, tuck_right: bool = False) -> None:
        with self._step_cv:
            target_seq = self._step_seq + 5
        with self._sim_lock:
            self._active_raw = np.asarray(raw, dtype=np.float32).copy()
            self._tuck_left = bool(tuck_left)
            self._tuck_right = bool(tuck_right)
        # Wait exact 5 sim steps (50 * 2 ms = 100 ms) synced to physics thread, or run synchronously if headless
        if self._worker_thread is None:
            self.step(SUBSTEPS_PER_COG_STEP)
            return
        deadline = time.perf_counter() + 1.5
        with self._step_cv:
            while (
                self._step_seq < target_seq
                and not self._abort_flag
                and self.power_on
                and not self._worker_stop.is_set()
            ):
                rem = deadline - time.perf_counter()
                if rem <= 0.0:
                    break
                self._step_cv.wait(timeout=rem)

    def _ensure_upright(self, timeout_s: float = 18.0) -> bool:
        """Wait for (or run synchronously) autonomous stand-up RL policy if G1 is fallen."""
        with self._sim_lock:
            fell = self.sys0.fell()
            rec = self.sys0.recovering
            if fell and not rec and self.power_on and not self.standby_mode:
                rec = self.sys0.start_standup()
                if rec:
                    self.log_event("warn", "[System 0] Обнаружено падение -> запуск автономной политики вставания...")
        if not fell and not rec:
            return True

        self.state = "RECOVERING"
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if not self.power_on or self.standby_mode:
                return False
            with self._sim_lock:
                still_rec = self.sys0.recovering
                still_fell = self.sys0.fell()
                # Re-arm standup automatically if cooldown expired after an incomplete recovery cycle
                if (not still_rec) and still_fell and self.sys0.recover_cooldown <= 0:
                    still_rec = self.sys0.start_standup()
            if not still_rec and not still_fell:
                return True
            if self._worker_thread is None:
                self.step(10)
            else:
                self._step_event.clear()
                self._step_event.wait(timeout=0.05)

        with self._sim_lock:
            return not self.sys0.fell()

    def step(self, n_substeps: int = 10) -> None:
        """Synchronous physics advance helper (used both by background loop and unit tests)."""
        started_msg = ""
        finished_msg = ""
        with self._sim_lock:
            was_rec = self.sys0.recovering
            if (
                self.power_on
                and not self.standby_mode
                and not self.sys0.flipping
                and not was_rec
                and self.sys0.recover_cooldown <= 0
                and self.sys0.fell()
                and self.sys0.start_standup()
            ):
                was_rec = True
                started_msg = "[System 0] Обнаружено падение -> автономное вставание (SuperMimic RL)"

            for _ in range(n_substeps):
                self.sys0.step_physics(
                    self._active_raw,
                    power_on=self.power_on,
                    tuck_left=self._tuck_left,
                    tuck_right=self._tuck_right,
                )

            now_rec = self.sys0.recovering
            if now_rec and not self._executing:
                self.state = "RECOVERING"
            elif was_rec and not now_rec:
                if not self.sys0.fell():
                    finished_msg = "[System 0] Робот самостоятельно встал на ноги!"
                else:
                    finished_msg = "[System 0] Цикл вставания завершён, требуется повторная стабилизация."
                if not self._executing and self.power_on and not self.standby_mode:
                    self.state = "IDLE"

            now = time.perf_counter()
            if (now - self._last_render_ts) >= PREVIEW_INTERVAL:
                self._render_sync()
                self._last_render_ts = now
            self._step_event.set()

        with self._step_cv:
            self._step_seq += 1
            self._step_cv.notify_all()

        if started_msg:
            self.log_event("warn", started_msg)
        if finished_msg:
            self.log_event("info" if "встал" in finished_msg else "warn", finished_msg)

    # ------------------------------------------------------------------ Cognitive / SLEEP workers
    def _tick_sys1_counter(self) -> None:
        self.sys1_count += 1

    def _propose_sys1(self, head_rgb: np.ndarray, proprio: np.ndarray, command: str) -> Tuple[np.ndarray, float]:
        chunk, sigma, _ = self._infer_sys1(head_rgb, proprio, command)
        return chunk, sigma

    def _infer_sys1(self, head_rgb: np.ndarray, proprio: np.ndarray, command: str) -> Tuple[np.ndarray, float, float]:
        img_small = np.array(
            Image.fromarray(head_rgb).resize((SLEEP_IMG_SIZE, SLEEP_IMG_SIZE), Image.BILINEAR),
            dtype=np.uint8,
            copy=True,
        )
        img_t = torch.from_numpy(img_small).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        st_t = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)
        tok_t = torch.tensor([tokenize(command, vocab=self.vla.vocab)], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        with self._net_lock:
            self.vla.eval()
            with torch.no_grad():
                feat = self.vla.encode(img_t, st_t, tok_t)
                chunk, sigma = self.vla.predict_action(feat, greedy=True)
        ms = (time.perf_counter() - t0) * 1000.0
        return chunk[0].detach().cpu().numpy(), float(sigma), ms

    def _run_sys1_closed_loop(self, command: str, init_chunk: np.ndarray, *, verify: bool = False) -> bool:
        """Execute trained System 1 chunks at 10 Hz with live uncertainty & outcome verification."""
        self.state = "SYS1"
        before_rgb = self._get_head_rgb_blocking().copy()
        with self._sim_lock:
            p0_xy = self.data.xpos[self.sys0.pelvis_id, :2].copy()
            p0_z = float(self.data.xpos[self.sys0.pelvis_id, 2])
            rhand0 = self.sys0.hook_pos("right").copy()
        max_dz = 0.0
        max_dr = 0.0
        chunk = init_chunk
        for cycle in range(14):
            with self._sim_lock:
                fell = self.sys0.fell()
            if self._abort_flag or not self.power_on or fell:
                return False
            # Execute first 3 steps of chunk (0.3s horizon), then re-evaluate vision & sigma
            for k in range(min(3, len(chunk))):
                with self._sim_lock:
                    fell = self.sys0.fell()
                if self._abort_flag or not self.power_on or fell:
                    return False
                raw = norm_to_raw_action(chunk[k])
                tuck_r = float(np.max(np.abs(raw[5:12] - default_raw_action()[5:12]))) < 0.30
                self._step_cog_from_worker(raw, tuck_left=True, tuck_right=tuck_r)
                self.sys1_count += 1
                with self._sim_lock:
                    max_dz = max(max_dz, abs(float(self.data.xpos[self.sys0.pelvis_id, 2]) - p0_z))
                    max_dr = max(max_dr, float(np.linalg.norm(self.sys0.hook_pos("right") - rhand0)))

            if cycle < 13:
                head_rgb = self._get_head_rgb_blocking()
                with self._sim_lock:
                    proprio = extract_proprio(
                        self.data, self.sys0.pelvis_id, self.sys0.torso_id, self.sys0.wrist_r_id, self.sys0.qadr, 0.0
                    )
                chunk, sigma, ms = self._infer_sys1(head_rgb, proprio, command)
                self.current_uncertainty = sigma
                self.latency_ms = ms
                if sigma > self.tau_think * 1.45:
                    self.log_event("warn", f"[Confidence Gate] σ выросла до {sigma:.3f} -> эскалация в System 2")
                    return False

        # Settle gently after Sys1 execution until leg gait fully hands back to stand PD
        stand = default_raw_action()
        for k in range(12):
            if self._abort_flag or not self.power_on:
                break
            self._step_cog_from_worker(stand, tuck_left=True, tuck_right=True)
            with self._sim_lock:
                holding = self.sys0.walk is None or self.sys0.walk.holding
            if k >= 9 and holding:
                break

        if verify:
            with self._sim_lock:
                d_walk = float(np.linalg.norm(self.data.xpos[self.sys0.pelvis_id, :2] - p0_xy))
            after_rgb = self._get_head_rgb_blocking().copy()
            verdict = self.planner.judge(before_rgb, after_rgb, command)
            # If VLM confirms goal wasn't reached and robot didn't execute a pronounced body skill
            if not verdict.get("done") and (d_walk < 0.18 and max_dz < 0.08 and max_dr < 0.14):
                what = verdict.get("what") or "нет видимого эффекта"
                self.log_event("warn", f"[Confidence Gate] System 1 не выполнил «{command}» ({what}) -> эскалация в System 2")
                return False

        self.log_event("sys1", f"[System 1] Навык «{command}» выполнен автономно (σ={self.current_uncertainty:.3f}).")
        return True

    def _run_flip(self, command: str) -> None:
        """Execute pretrained ONNX backflip policy under System 0 control."""
        with self._sim_lock:
            started = self.sys0.start_flip()
        if not started:
            self.log_event("warn", "[System 0] Политика сальто (models/g1_flip) не загружена.")
            return
        self.state = "SYS1"
        t0 = time.perf_counter()
        self.log_event("sys1", f"[System 0] Запуск политики сальто «{command}» (wbc-mjlab ONNX)...")
        deadline = time.perf_counter() + 6.0
        while time.perf_counter() < deadline:
            if not self.power_on or self.standby_mode or self._abort_flag:
                with self._sim_lock:
                    self.sys0.flipping = False
                return
            with self._sim_lock:
                still = self.sys0.flipping
            if not still:
                break
            if self._worker_thread is None:
                self.step(10)
            else:
                self._step_event.clear()
                self._step_event.wait(timeout=0.05)

        # Let landing settle briefly, then auto-recover if landing caused a fall
        for _ in range(15):
            if not self.power_on or self.standby_mode:
                break
            if self._worker_thread is None:
                self.step(10)
            else:
                self._step_event.clear()
                self._step_event.wait(timeout=0.05)
        with self._sim_lock:
            fell_after = self.sys0.fell()
        if fell_after and self.power_on and not self.standby_mode:
            self.log_event("warn", "[System 0] После приземления требуется вставание — запускаю SuperMimic RL...")
            self._ensure_upright()
        else:
            self.log_event("sys1", "[System 0] Сальто выполнено!")
        self.latency_ms = (time.perf_counter() - t0) * 1000.0

    def _execute_command(self, command: str) -> None:
        self._abort_flag = False
        self._executing = True
        self._last_activity = time.time()
        try:
            with self._sim_lock:
                need_standup = self.sys0.fell() or self.sys0.recovering
            if need_standup:
                if not self._ensure_upright():
                    self.log_event("warn", "[System 0] Не удалось встать перед выполнением команды.")
                    with self._cmd_cv:
                        if self._active_goal == command:
                            self._clear_goal_locked(clear_pending=False)
                    return

            if wants_flip(command):
                with self._cmd_cv:
                    if self._active_goal == command:
                        self._clear_goal_locked(clear_pending=False)
                self._run_flip(command)
                return

            head_rgb = self._get_head_rgb_blocking()
            with self._sim_lock:
                proprio = extract_proprio(
                    self.data, self.sys0.pelvis_id, self.sys0.torso_id, self.sys0.wrist_r_id, self.sys0.qadr, 0.0
                )
            chunk, raw_sigma, ms = self._infer_sys1(head_rgb, proprio, command)
            sigma = self._effective_sigma(command, raw_sigma)
            skill_key = self._skill_key(command)
            skill_known = skill_key in self.learner.consolidated_skills
            self.current_uncertainty = sigma
            self.latency_ms = ms

            # Confidence Gate: only use System 1 if THIS skill is consolidated AND sigma <= tau_think
            if skill_known and sigma <= self.tau_think and self.vla.cms.used_slots() > 1:
                self.log_event(
                    "sys1",
                    f"[System 1] σ={sigma:.3f} <= τ={self.tau_think:.2f} · {ms:.1f} мс · Автопилот по памяти CMS!",
                )
                if self._run_sys1_closed_loop(command, chunk, verify=True):
                    with self._cmd_cv:
                        if self._active_goal == command:
                            self._clear_goal_locked(clear_pending=False)
                    return

            # Fallback / Deliberative System 2 (VLM + Locomotion + Whole-Body IK)
            self.state = "SYS2"
            self.sys2_count += 1
            lessons = self.learner.get_lessons(skill_key)
            if lessons:
                self.log_event("info", f"[Память уроков] Использую {len(lessons)} прошлый(х) урок(а) для «{command}».")
            if not skill_known:
                self.log_event(
                    "sys2",
                    f"[System 2] Навык «{command}» ещё не изучен (σ={sigma:.3f}) · Запуск VLM ({self.planner.active_backend})...",
                )
            else:
                self.log_event(
                    "sys2",
                    f"[System 2] σ={sigma:.3f} > τ={self.tau_think:.2f} · Запуск VLM ({self.planner.active_backend})...",
                )
            t0 = time.perf_counter()
            res = self.sys2.execute(command, lessons=lessons)
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            samples = res.get("samples") or []
            if res.get("ok"):
                with self._cmd_cv:
                    if self._active_goal == command:
                        self._clear_goal_locked(clear_pending=False)
                if samples:
                    tid = self._task_id(command)
                    for rgb_128, pr, cmd, ch in samples:
                        self.wake_buffer.append((rgb_128, pr, cmd, ch, tid))
                    self._pending_skills.add(skill_key)
                    self.unconsolidated_steps += len(samples)
                    self._sleep_done_for_batch = False
                    self._run_wake_update(steps=2, allow_consolidate=False)
                    self.log_event(
                        "info",
                        f"[Wake] Сохранено {len(samples)} траекторных чанков «{command}» в буфер памяти.",
                    )
                elif res.get("already_done"):
                    self.log_event("info", f"[System 2] Цель «{command}» уже выполнена — завершаю задачу.")
            else:
                self.log_event("warn", f"[System 2] Команда «{command}» не завершена ({res.get('reason', 'unknown')}).")
                self._handle_failed_episode(command, skill_key, head_rgb, proprio, chunk, samples, res.get("failure_info"))
                self._maybe_schedule_retry(command, str(res.get("reason", "")))
        except Exception as exc:
            self.log_event("error", f"Ошибка выполнения «{command}»: {exc}")
        finally:
            self._active_raw = default_raw_action()
            self._tuck_left = True
            self._tuck_right = True
            self._executing = False
            self._last_activity = time.time()
            if not self.power_on:
                self.state = "OFF"
            elif self.standby_mode:
                self.state = "STANDBY"
            elif self.sys0.recovering:
                self.state = "RECOVERING"
            else:
                self.state = "IDLE"

    def _handle_failed_episode(
        self,
        command: str,
        skill_key: str,
        head_rgb: np.ndarray,
        proprio: np.ndarray,
        fallback_chunk: np.ndarray,
        samples: list,
        failure_info: Optional[Dict[str, Any]],
    ) -> None:
        """Store negative episode context, extract a VLM episodic lesson, stand back up, and run failure REM sleep."""
        # If robot fell during failed episode, autonomously recover upright stance first
        with self._sim_lock:
            fell_now = self.sys0.fell() or self.sys0.recovering
        if fell_now and self.power_on and not self.standby_mode:
            self._ensure_upright()

        tid = self._task_id(command)
        if samples:
            for rgb_128, pr, cmd, ch in samples:
                self.failure_buffer.append((rgb_128, pr, cmd, ch, tid))
        else:
            rgb_small = np.array(
                Image.fromarray(head_rgb).resize((SLEEP_IMG_SIZE, SLEEP_IMG_SIZE), Image.BILINEAR),
                dtype=np.uint8,
                copy=True,
            )
            self.failure_buffer.append((rgb_small, proprio.copy(), command, fallback_chunk.copy(), tid))
        self._unlearned_failures += 1

        if failure_info:
            lesson = self.planner.reflect_failure(
                failure_info.get("before_rgb"),
                failure_info.get("after_rgb"),
                command,
                failure_info,
            )
            if lesson:
                self.learner.add_lesson(skill_key, lesson)
                self.log_event("sleep", f"[Урок эпизода] {lesson}")

        if self.power_on and not self.standby_mode:
            self._run_failure_sleep()

    def _maybe_schedule_retry(self, command: str, reason: str) -> None:
        """Re-queue an unfinished goal after failure sleep while retries remain."""
        non_retryable = reason in ("VLM недоступна", "Остановлено")
        with self._cmd_cv:
            if self._active_goal != command or self._pending_cmd is not None:
                return
            if non_retryable or not self.power_on or self.standby_mode or self._abort_flag:
                if non_retryable or not self.power_on or self._abort_flag:
                    self._clear_goal_locked(clear_pending=False)
                return
            if self._goal_retries_left <= 0:
                self._clear_goal_locked(clear_pending=False)
                self.log_event(
                    "warn",
                    f"[Авто-продолжение] Лимит попыток для «{command}» исчерпан — останавливаюсь.",
                )
                return
            self._goal_retries_left -= 1
            attempt_num = (self._max_post_sleep_retries - self._goal_retries_left) + 1
            total_attempts = self._max_post_sleep_retries + 1
            self._pending_cmd = command
            self.current_command = command
            self._cmd_cv.notify_all()
        self.log_event(
            "info",
            f"[Авто-продолжение] После сна повторяю «{command}» с учётом урока (попытка {attempt_num}/{total_attempts})...",
        )

    def _run_failure_sleep(self) -> None:
        """Automatic post-failure REM sleep: explore away from failed trajectories without polluting System 1."""
        if not self.failure_buffer:
            return
        self.state = "MICRO_SLEEP"
        self.total_sleep_cycles += 1
        t0 = time.perf_counter()
        self.log_event(
            "sleep",
            f"[Авто-сон (неудача) #{self.total_sleep_cycles}] Анализ провальной траектории (чанков={len(self.failure_buffer)})...",
        )
        sel = list(self.failure_buffer)[-12:]
        imgs = torch.from_numpy(np.stack([x[0] for x in sel])).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        sts = torch.from_numpy(np.stack([x[1] for x in sel])).float().to(self.device)
        toks = torch.tensor([tokenize(x[2], vocab=self.vla.vocab) for x in sel], dtype=torch.long, device=self.device)
        ys = torch.from_numpy(np.stack([x[3] for x in sel])).float().to(self.device)

        with self._net_lock:
            metrics = self.learner.rem_failure_sleep((imgs, sts, toks, ys), steps=3)
            self.learner.save_checkpoint(self.ckpt_path)

        # Keep epistemic uncertainty >= 0.85 for unconsolidated failed skills
        self.current_uncertainty = max(0.85, float(metrics.get("sigma", 0.85)))
        self._unlearned_failures = 0
        self.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.log_event("sleep", "[Авто-сон (неудача)] Урок и REM-коррекция L0 сохранены (навык остаётся в System 2).")
        self.state = "IDLE" if self.power_on else "OFF"

    def _sample_wake_tensors(self, batch_sz: int):
        buf = list(self.wake_buffer)
        sel = random.sample(buf, min(batch_sz, len(buf)))
        imgs = torch.from_numpy(np.stack([x[0] for x in sel])).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        sts = torch.from_numpy(np.stack([x[1] for x in sel])).float().to(self.device)
        toks = torch.tensor([tokenize(x[2], vocab=self.vla.vocab) for x in sel], dtype=torch.long, device=self.device)
        ys = torch.from_numpy(np.stack([x[3] for x in sel])).float().to(self.device)
        tids = torch.tensor([x[4] for x in sel], dtype=torch.long, device=self.device)
        return imgs, sts, toks, ys, tids

    def _run_wake_update(self, steps: int = 1, *, allow_consolidate: bool = False) -> float:
        if len(self.wake_buffer) < 4:
            return 0.0
        loss_val = 0.0
        with self._net_lock:
            for _ in range(steps):
                batch = self._sample_wake_tensors(16)
                loss_val = self.learner.wake_step(batch, allow_consolidate=allow_consolidate)
        return loss_val

    def _run_micro_sleep(self) -> None:
        if len(self.wake_buffer) < 6:
            return
        self.state = "MICRO_SLEEP"
        t0 = time.perf_counter()
        loss_val = self._run_wake_update(steps=2, allow_consolidate=False)
        with self._net_lock:
            img, st, tok, y, _ = self._sample_wake_tensors(min(16, len(self.wake_buffer)))
            with torch.no_grad():
                feat = self.vla.encode(img, st, tok)
                mu, _ = self.vla.dists(feat, y)
                mse = float(F.mse_loss(mu, y).item())
                _, sigma = self.vla.predict_action(feat[:1], greedy=True)
        self.current_uncertainty = sigma
        self.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.unconsolidated_steps = max(0, self.unconsolidated_steps - 8)
        self.log_event("sleep", f"[Микро-сон L0] MSE={mse:.4f} · σ={sigma:.3f} · loss={loss_val:.2f}")
        # Automatically consolidate episode batch on idle if policy converged cleanly
        if self._pending_skills and self.unconsolidated_steps == 0 and mse < 0.025:
            self._run_deep_sleep(as_micro=True)
        self.state = "IDLE" if self.power_on else "OFF"

    def _run_deep_sleep(self, *, as_micro: bool = False) -> None:
        if self._sleep_done_for_batch or len(self.wake_buffer) < 6:
            return
        self.state = "MICRO_SLEEP" if as_micro else "STANDBY"
        self.total_sleep_cycles += 1
        tag = "Авто-сон эпизода" if as_micro else "Stand-by Сон"
        self.log_event("sleep", f"[{tag} #{self.total_sleep_cycles}] NREM консолидация + REM сновидения...")
        t0 = time.perf_counter()

        # 1. Warm short-term Wake buffer for consolidation
        n_warm = min(4, max(1, len(self.wake_buffer) // 6))
        self._run_wake_update(steps=n_warm, allow_consolidate=False)

        with self._net_lock:
            val_img, val_st, val_tok, val_y, _ = self._sample_wake_tensors(min(12, len(self.wake_buffer)))

            def tau_score(m) -> float:
                with torch.no_grad():
                    f = m.encode(val_img, val_st, val_tok)
                    mu, _ = m.dists(f, val_y)
                    return -float(F.mse_loss(mu, val_y).item())

            # 2+3. Single clean SLEEP boundary: NREM consolidate L0->L1->L2, REM dream, anchors + prev_teacher
            ctx_img, ctx_st, ctx_tok = val_img[:1], val_st[:1], val_tok[:1]
            self.learner.end_of_task_sleep(context=(ctx_img, ctx_st, ctx_tok), tau=tau_score)
            self.learner.consolidated_skills.update(self._pending_skills)
            self._pending_skills.clear()

            # Recompute post-sleep uncertainty on stored trajectories
            with torch.no_grad():
                f_after = self.vla.encode(val_img[:1], val_st[:1], val_tok[:1])
                _, sigma_after = self.vla.predict_action(f_after, greedy=True)

            self.learner.save_checkpoint(self.ckpt_path)

        self.current_uncertainty = sigma_after
        self.unconsolidated_steps = 0
        self._sleep_done_for_batch = True
        self.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.log_event(
            "sleep",
            f"[Сон завершён] Слоты CMS={self.vla.cms.used_slots()}/{self.vla.cms.total_slots()} · σ={sigma_after:.3f} · Чекпоинт сохранён",
        )

    def _cognitive_worker(self) -> None:
        while not self._worker_stop.is_set():
            cmd: Optional[str] = None
            with self._cmd_cv:
                if self._pending_cmd is not None:
                    cmd = self._pending_cmd
                    self._pending_cmd = None
                else:
                    self._cmd_cv.wait(timeout=0.4)
                    if self._pending_cmd is not None:
                        cmd = self._pending_cmd
                        self._pending_cmd = None

            if self._worker_stop.is_set():
                break

            if cmd is not None:
                self._execute_command(cmd)
                continue

            if not self.power_on:
                self.state = "OFF"
                continue

            if self.standby_mode:
                self.state = "STANDBY"
                try:
                    self._run_deep_sleep()
                except Exception as exc:
                    self.log_event("error", f"Ошибка Deep Sleep: {exc}")
                    self._sleep_done_for_batch = True
                continue

            idle_s = time.time() - self._last_activity
            if (not self._executing) and idle_s > MICRO_SLEEP_IDLE_S and self.unconsolidated_steps >= MICRO_SLEEP_MIN_STEPS:
                try:
                    self._run_micro_sleep()
                except Exception as exc:
                    self.log_event("error", f"Ошибка Micro-Sleep: {exc}")
                    self.unconsolidated_steps = 0
                    self.state = "IDLE" if self.power_on else "OFF"

    # ------------------------------------------------------------------ Physics Loop & Telemetry
    def loop(self, stop_event: threading.Event) -> None:
        """Main background loop: starts cognitive worker and steps 500 Hz MuJoCo physics at 50 Hz cadence."""
        self._loop_thread_id = threading.get_ident()
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._cognitive_worker, daemon=True)
        self._worker_thread.start()

        with self._sim_lock:
            self._render_sync()

        step_dt = 10 * PHYSICS_DT  # 0.02 s (50 Hz outer cadence, 500 Hz inner integrator)
        try:
            while not stop_event.is_set():
                t0 = time.perf_counter()
                try:
                    self.step(10)
                except Exception as exc:
                    self.log_event("error", f"Ошибка цикла физики/рендера: {exc}")
                    time.sleep(0.1)
                    continue
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.001, step_dt - elapsed))
        finally:
            self._worker_stop.set()
            with self._cmd_cv:
                self._cmd_cv.notify_all()
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=2.0)
            self._close_renderers()

    def telemetry(self) -> Dict[str, Any]:
        with self._sim_lock:
            pelvis_z = float(self.data.xpos[self.sys0.pelvis_id, 2])
            tilt = float(self.data.xmat[self.sys0.torso_id].reshape(3, 3)[2, 2])
            fell = self.sys0.fell()
        total = max(1, self.sys1_count + self.sys2_count)
        return {
            "state": self.state,
            "power_on": self.power_on,
            "standby_mode": self.standby_mode,
            "recovering": self.sys0.recovering,
            "command": self.current_command,
            "vlm_backend": self.planner.active_backend,
            "pi0_phase": self.pi0.phase(),
            "uncertainty": float(self.current_uncertainty),
            "tau_think": float(self.tau_think),
            "latency_ms": float(self.latency_ms),
            "pelvis_z": pelvis_z,
            "tilt": tilt,
            "fell": fell,
            "cms_slots_used": self.vla.cms.used_slots(),
            "cms_slots_total": self.vla.cms.total_slots(),
            "wake_buffer_len": len(self.wake_buffer),
            "failure_buffer_len": len(self.failure_buffer),
            "lessons_count": sum(len(v) for v in self.learner.skill_lessons.values()),
            "unconsolidated_steps": self.unconsolidated_steps,
            "sleep_cycles": self.total_sleep_cycles,
            "sys1_rate": 100.0 * self.sys1_count / total,
            "sys2_rate": 100.0 * self.sys2_count / total,
            "outcome": "recovering" if self.sys0.recovering else ("fall" if fell else "ok"),
        }

    def _close_renderers(self) -> None:
        if self._preview_renderer is not None:
            try:
                self._preview_renderer.close()
            except Exception:
                pass
            self._preview_renderer = None
        if self._head_renderer is not None:
            try:
                self._head_renderer.close()
            except Exception:
                pass
            self._head_renderer = None

    def close(self) -> None:
        self._worker_stop.set()
        with self._cmd_cv:
            self._cmd_cv.notify_all()
        with self._step_cv:
            self._step_cv.notify_all()
        if self._worker_thread is None:
            self._close_renderers()


RobotEngine = CognitiveRobotEngine
