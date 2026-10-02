"""Whole-body e2e teacher: one VLA → absolute body joints → Joint-PD.

OpenHLM-style contract for this repo:
  language + head RGB + proprio(29) → action chunk → q_body[29]
  Dex3 fingers stay on the curl PD (not in the base π0/π0.5 chunk).
  Low-level Joint-PD (+ light balance prior) tracks q_body at physics rate.

VRAM rule: only this teacher may own the GPU. L2/student stay on CPU.
CPU load is refused — it freezes the 500 Hz loop.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

import numpy as np
import torch

from agent.h2 import N_ACT, STAND_Q
from agent.policy import resolve_pi0_device
from agent.vla import (
    BLEND_HOLDS,
    HOLD_TICKS,
    PI0_REPO,
    SIM_DT,
    TORCH_GATE,
    VLA_H,
    VLA_W,
    _import_pi0,
    _load_file_nommap,
    _pi0_tokenizer_available,
)

# Prefer π0.5; fall back to pi0_base if the local cache only has that.
E2E_REPO_DEFAULT = "lerobot/pi05_base"
BODY_DELTA = 0.10  # rad per chunk step (normalized action → joint delta)
E2E_BALANCE = 0.35  # keep a fraction of IPM prior under the VLA pose
_BASE_IMAGE = "observation.images.base_0_rgb"
_STATE = "observation.state"


def e2e_enabled() -> bool:
    return os.getenv("E2E_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def resolve_e2e_device(spec: str | None = None) -> torch.device:
    raw = spec if spec is not None else os.getenv("E2E_DEVICE", "")
    raw = (raw or os.getenv("PI0_DEVICE", "") or "").lower().strip()
    if not raw:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return resolve_pi0_device(raw)


def e2e_repo() -> str:
    return (os.getenv("E2E_REPO") or E2E_REPO_DEFAULT).strip()


def retarget_body(action, q_body, lo, hi) -> np.ndarray:
    """Normalized VLA action → absolute 29-D body targets anchored at current q."""
    q = np.asarray(q_body, dtype=np.float32).reshape(N_ACT)
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    delta = np.zeros(N_ACT, dtype=np.float32)
    n = min(N_ACT, int(raw.shape[0]))
    delta[:n] = np.clip(raw[:n], -1.0, 1.0) * np.float32(BODY_DELTA)
    out = q + delta
    return np.clip(
        out,
        np.asarray(lo, dtype=np.float32).reshape(N_ACT),
        np.asarray(hi, dtype=np.float32).reshape(N_ACT),
    )


class E2ETeacher:
    """Lazy whole-body VLA. `command` returns None until weights + first chunk exist."""

    def __init__(self, device: torch.device):
        self.device = device
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._policy = None
        self._pre = None
        self._post = None
        self._ready = False
        self._busy = False
        self._error = ""
        self._pending: str | None = None
        self._gen = 0
        self._chunk: np.ndarray | None = None
        self._i = 0
        self._hold = 0
        self._anchor: np.ndarray | None = None
        self._latched: np.ndarray | None = None
        self._queued: np.ndarray | None = None
        self._infer_s: float | None = None
        self._slow_noted = False
        self._repo = e2e_repo()

    @property
    def error(self) -> str:
        return self._error

    def phase(self) -> str:
        with self._lock:
            if self._error and not self._ready:
                return "error"
            if self._thread is None:
                return "off"
            if not self._ready:
                return "loading"
            if self._chunk is not None or self._latched is not None:
                return "run"
            return "ready"

    def infer_s(self) -> float | None:
        with self._lock:
            return self._infer_s

    def load_now(self) -> str | None:
        with self._lock:
            if self._ready or self._thread is not None or self._error:
                note = self._pending
                self._pending = None
                return note
        if str(self.device).startswith("cpu"):
            with self._lock:
                self._error = (
                    "e2e на CPU слишком медленная для sim-loop. "
                    "Поставь E2E_DEVICE=cuda (освободи VRAM) или E2E_ENABLE=0."
                )
                self._pending = f"E2E пропущена: {self._error}"
                note = self._pending
                self._pending = None
                return note
        try:
            with TORCH_GATE:
                self._load_weights()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._pending = f"E2E не поднялась: {self._error}"
                note = self._pending
                self._pending = None
                return note
        with self._lock:
            self._ready = True
            self._pending = f"E2E готова ({self._repo} @ {self.device})."
            self._thread = threading.Thread(target=self._serve, name="e2e", daemon=True)
            self._thread.start()
            note = self._pending
            self._pending = None
            return note

    def wants_still_frame(self) -> bool:
        with self._lock:
            if not self._ready or self._busy or self._queued is not None:
                return False
            return self._chunk is None

    def submit(self, rgb: np.ndarray, text: str, state: np.ndarray) -> None:
        with self._lock:
            if not self._ready or self._busy or self._queued is not None:
                return
            if self._chunk is not None:
                return
            self._busy = True
            gen = int(self._gen)
        try:
            self._jobs.put_nowait(
                (
                    np.ascontiguousarray(rgb).copy(),
                    str(text or ""),
                    np.asarray(state, dtype=np.float32).copy(),
                    gen,
                )
            )
        except queue.Full:
            with self._lock:
                self._busy = False

    def command(self, q_body, lo, hi) -> np.ndarray | None:
        """Next absolute 29-D body target, or None while waiting for the first chunk."""
        with self._lock:
            self._install_queued()
            if self._chunk is None:
                if self._latched is None:
                    return None
                return self._latched.copy()
            if self._anchor is None:
                self._anchor = np.asarray(q_body, dtype=np.float32).reshape(N_ACT).copy()
            q = retarget_body(self._chunk[self._i], self._anchor, lo, hi)
            self._latched = q
            self._hold += 1
            if self._hold >= HOLD_TICKS:
                self._hold = 0
                self._i += 1
                self._anchor = None
                if self._i >= len(self._chunk):
                    self._chunk = None
                    self._i = 0
                    self._install_queued()
            return q.copy()

    def reset_chunk(self) -> None:
        with self._lock:
            self._gen += 1
            self._chunk = None
            self._i = 0
            self._hold = 0
            self._anchor = None
            self._latched = None
            self._queued = None
            self._busy = False
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

    def _install_queued(self) -> None:
        if self._queued is None or self._chunk is not None:
            return
        self._chunk = self._queued
        self._queued = None
        self._i = 0
        self._hold = 0
        self._anchor = None

    def _serve(self) -> None:
        while True:
            rgb, text, state, gen = self._jobs.get()
            try:
                with self._lock:
                    current = int(self._gen)
                if gen != current:
                    continue
                chunk, elapsed = self._infer(rgb, text, state)
                with self._lock:
                    self._infer_s = float(elapsed)
                    chunk_s = float(len(chunk)) * HOLD_TICKS * SIM_DT
                    if elapsed > chunk_s and not self._slow_noted:
                        self._slow_noted = True
                        self._pending = (
                            f"E2E: инференс {elapsed:.2f} с длиннее чанка {chunk_s:.2f} с."
                        )
                    if gen == int(self._gen):
                        self._queued = chunk
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._pending = f"E2E: {self._error}"
            finally:
                with self._lock:
                    self._busy = False

    def _load_weights(self) -> None:
        if not _pi0_tokenizer_available():
            raise RuntimeError(
                "нет HF_TOKEN и локального google/paligemma-3b-pt-224 — e2e пропущена."
            )
        policy_cls, make_processors = _import_pi0()
        import safetensors.torch as safetensors_torch

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        safetensors_torch.load_file = _load_file_nommap
        repo = self._repo
        try:
            policy = policy_cls.from_pretrained(repo, local_files_only=True)
        except Exception:
            if repo != PI0_REPO:
                repo = PI0_REPO
                self._repo = repo
                policy = policy_cls.from_pretrained(repo, local_files_only=True)
            else:
                raise
        policy.eval()
        policy.to(self.device)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            pre, post = make_processors(
                policy.config,
                repo,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        except Exception as exc:
            msg = str(exc).lower()
            gated = any(s in msg for s in ("401", "gated", "paligemma", "unauthorized", "auth"))
            if gated and not token:
                raise RuntimeError(
                    "e2e processors need HF_TOKEN + google/paligemma-3b-pt-224 license."
                ) from exc
            raise
        self._policy = policy
        self._pre = pre
        self._post = post

    def _infer(self, rgb: np.ndarray, text: str, state: np.ndarray) -> tuple[np.ndarray, float]:
        with TORCH_GATE:
            return self._infer_unlocked(rgb, text, state)

    def _infer_unlocked(self, rgb: np.ndarray, text: str, state: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        # Proprio for e2e is the 29 body hinges (OpenHLM-style), not the 18-D cmd bus.
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(-1))
        if int(state_t.numel()) < N_ACT:
            pad = torch.zeros(N_ACT, dtype=torch.float32)
            pad[: int(state_t.numel())] = state_t
            state_t = pad
        elif int(state_t.numel()) > N_ACT:
            state_t = state_t[:N_ACT]
        obs = {
            _BASE_IMAGE: image,
            _STATE: state_t,
            "task": text or "perform the instructed whole-body task",
        }
        batch = self._pre(obs)
        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(batch)
        try:
            chunk = self._post(chunk)
        except Exception:
            pass
        arr = np.asarray(chunk.detach().float().cpu().numpy(), dtype=np.float32)
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] < N_ACT:
            pad = np.zeros((arr.shape[0], N_ACT), dtype=np.float32)
            pad[:, : arr.shape[-1]] = arr
            arr = pad
        return arr, time.perf_counter() - t0


def stand_body() -> np.ndarray:
    return np.asarray(STAND_Q, dtype=np.float32).copy()
