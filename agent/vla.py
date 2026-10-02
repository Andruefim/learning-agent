"""System 1.5. π0 proposes a chunk; the walk net still owns the legs.

The checkpoint is lerobot/pi0_base (Gemma license). It is loaded on the first
interaction that names a pixel, on a side thread, so physics does not wait on
the download. The live loop asks for the next image only after the current
chunk has finished, so the arm is not given a new gesture while it is still
moving. While that result is late, forward and yaw speed decay; the arm
targets stay. The chunk itself is played as π0 produced it.
Wrist cameras are not added: missing views stay masked.

Its 32-D normalized action is retargeted into the existing 18-D command.
Dims 0:14 are arm deltas in actuator order. Dims 14 and 16 are forward and
yaw speed. Lateral speed stays 0 because this gait falls when it strafes.
Height stays at the stand. Fingers are not in the chunk.
"""

from __future__ import annotations

import json
import math
import queue
import struct
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch

from agent.config import STAND_Z
from agent.h2 import colliding_geoms
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, CMD_VY, CMD_WZ, clip_command, stand_command

# The sim thread and this loader both enter torch. On this build that overlap
# is an access violation, so only one of them runs native torch at a time.
TORCH_GATE = threading.Lock()

PI0_REPO = "lerobot/pi0_base"
# After approach, if the first chunk never arrives, fall back to geometric IK.
PI0_WAIT_S = 3.0
VLA_H, VLA_W = 240, 320
ARM_DELTA = 0.12
BASE_VX = 0.25
BASE_WZ = 0.40
BASE_DEAD = 0.15
HOLD_TICKS = 10
SIM_DT = 0.002
LATCH_TAU_S = 0.25
BLEND_HOLDS = 2
WORLD_MOVE_M = 0.02
HINGE_LEVER_M = 0.30
_BASE_IMAGE = "observation.images.base_0_rgb"
_STATE = "observation.state"


def subtree_bodies(model, root: int) -> set[int]:
    root = int(root)
    out = {root}
    n = int(model.nbody)
    for i in range(n):
        b = int(i)
        guard = 0
        while b != 0 and guard < 64:
            if b == root:
                out.add(int(i))
                break
            b = int(model.body_parentid[b])
            guard += 1
    return out


def arm_contact_geoms(model) -> set[int]:
    geoms: set[int] = set()
    for name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link"):
        try:
            root = int(model.body(name).id)
        except KeyError:
            continue
        for body in subtree_bodies(model, root):
            geoms.update(colliding_geoms(model, body))
    return geoms


def pose_moved(x0, m0, x1, m1, bodies, min_m: float = WORLD_MOVE_M, lever: float = HINGE_LEVER_M) -> bool:
    """True when a contacted body translated or swung a point `lever` meters from its origin."""
    if x0 is None or m0 is None or not bodies:
        return False
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    for body in bodies:
        b = int(body)
        travel = float(np.linalg.norm(x1[b] - x0[b]))
        r0 = m0[b].reshape(3, 3)
        r1 = m1[b].reshape(3, 3)
        cosine = float(np.clip((np.trace(r0.T @ r1) - 1.0) * 0.5, -1.0, 1.0))
        swing = float(np.arccos(cosine)) * float(lever)
        if max(travel, swing) > float(min_m):
            return True
    return False


def _axis(value: float, scale: float) -> float:
    v = float(np.clip(value, -1.0, 1.0))
    if abs(v) < BASE_DEAD:
        return 0.0
    return float(np.clip(v * scale, -scale, scale))


def retarget_action(action, arm_q, lo, hi) -> np.ndarray:
    """One normalized π0 action → one 18-D command anchored at the current arms."""
    raw = np.zeros(32, dtype=np.float32)
    src = np.asarray(action, dtype=np.float32).reshape(-1)
    n = min(32, int(src.shape[0]))
    raw[:n] = src[:n]
    arms = np.asarray(arm_q, dtype=np.float32).reshape(14)
    delta = np.clip(raw[:14], -1.0, 1.0) * np.float32(ARM_DELTA)
    cmd = stand_command()
    cmd[CMD_ARMS] = np.clip(arms + delta, np.asarray(lo, dtype=np.float32).reshape(14), np.asarray(hi, dtype=np.float32).reshape(14))
    cmd[CMD_VX] = np.float32(_axis(float(raw[14]), BASE_VX))
    cmd[CMD_VY] = np.float32(0.0)
    cmd[CMD_WZ] = np.float32(_axis(float(raw[16]), BASE_WZ))
    cmd[CMD_H] = np.float32(STAND_Z)
    return clip_command(cmd)


def _bleed_base(cmd: np.ndarray) -> np.ndarray:
    """Ease vx/wz toward zero. Arm targets stay. One call is one sim tick."""
    out = np.array(cmd, dtype=np.float32, copy=True)
    keep = math.exp(-SIM_DT / LATCH_TAU_S)
    out[CMD_VX] = np.float32(float(out[CMD_VX]) * keep)
    out[CMD_WZ] = np.float32(float(out[CMD_WZ]) * keep)
    if abs(float(out[CMD_VX])) < 1e-3:
        out[CMD_VX] = np.float32(0.0)
    if abs(float(out[CMD_WZ])) < 1e-3:
        out[CMD_WZ] = np.float32(0.0)
    return out


def _torch_distributed_ok() -> bool:
    try:
        import torch.distributed.distributed_c10d  # noqa: F401
    except Exception:
        return False
    return True


_SAFETENSORS_DTYPE = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_safetensors(path) -> dict[str, torch.Tensor]:
    """Read a checkpoint without mmap.

    ``safetensors.torch.load_file`` maps the file and then calls ``.item()`` on
    that storage. On this torch build that faults, so the bytes are copied in.
    """
    path = Path(path)
    with path.open("rb") as handle:
        header_n = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_n))
        base = 8 + header_n
        out: dict[str, torch.Tensor] = {}
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            handle.seek(base + int(start))
            raw = handle.read(int(end) - int(start))
            tensor = torch.frombuffer(raw, dtype=_SAFETENSORS_DTYPE[spec["dtype"]])
            out[key] = tensor.reshape(spec["shape"]).clone()
        return out


def _load_file_nommap(filename, device="cpu"):
    tensors = read_safetensors(filename)
    if str(device) not in ("cpu", "cpu:0"):
        tensors = {key: value.to(device) for key, value in tensors.items()}
    return tensors


def _pi0_enabled() -> bool:
    import os

    return os.getenv("PI0_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _pi0_tokenizer_available() -> bool:
    """Processors need gated google/paligemma-3b-pt-224. Skip the crashy weight load if missing."""
    import os

    if not _pi0_enabled():
        return False
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        for name in ("tokenizer.model", "tokenizer.json"):
            path = try_to_load_from_cache("google/paligemma-3b-pt-224", name)
            if path is not None and path is not _CACHED_NO_EXIST and Path(path).is_file():
                return True
    except Exception:
        pass
    return False


def _import_pi0():
    """Import π0 without pulling every other policy.

    This torch build has no distributed extension. Importing ``lerobot.policies``
    loads those policies, and they import torchao, which needs that extension.
    """
    if not _torch_distributed_ok():
        import transformers

        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        if (major, minor) >= (5, 4):
            raise RuntimeError(
                "transformers>=5.4 imports torchao at startup, and this torch build has no "
                "distributed extension. Install transformers==5.2.0."
            )
        import transformers.utils.import_utils as iu

        iu.is_torchao_available = lambda: False
        import transformers.utils as utils

        utils.is_torchao_available = lambda: False
        import lerobot

        if "lerobot.policies" not in sys.modules:
            policies = types.ModuleType("lerobot.policies")
            policies.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
            policies.__package__ = "lerobot.policies"
            sys.modules["lerobot.policies"] = policies
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors

    return PI0Policy, make_pre_post_processors


class System15:
    """Lazy π0. `command` is safe before the weights exist: it returns None."""

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
        self._blend_from: np.ndarray | None = None
        self._blend_left = 0
        self._blend_total = 0
        self._infer_s: float | None = None
        self._slow_noted = False
        self._load_noted = False

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

    def load_now(self) -> str | None:
        """Load on the calling thread.

        safetensors faults (0xC0000005) in torch on a fresh thread with this build.
        The process that already runs the walk net can read the checkpoint.
        """
        with self._lock:
            if self._ready or self._thread is not None or self._error:
                note = self._pending
                self._pending = None
                return note
        # pi0_base on CPU blocks the physics loop for minutes (looks "stuck forever").
        if str(self.device).startswith("cpu"):
            with self._lock:
                self._error = (
                    "π0 на CPU слишком медленная для sim-loop. "
                    "Поставь PI0_DEVICE=cuda (освободи VRAM у Ollama) или PI0_ENABLE=0 для IK."
                )
                self._pending = f"System 1.5 пропущена: {self._error}"
                note = self._pending
                self._pending = None
                return note
        try:
            # Hold the gate: another torch entry on this build is an access violation.
            with TORCH_GATE:
                self._load_weights()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._pending = f"System 1.5 не поднялась: {self._error}"
                note = self._pending
                self._pending = None
                return note
        with self._lock:
            self._ready = True
            self._pending = f"System 1.5 готова ({self.device})."
            self._thread = threading.Thread(target=self._serve, name="pi0", daemon=True)
            self._thread.start()
            note = self._pending
            self._pending = None
            return note

    def ensure_started(self) -> str | None:
        with self._lock:
            if self._thread is None and not self._ready and not self._error and not self._load_noted:
                self._pending = "System 1.5: гружу lerobot/pi0_base."
                self._load_noted = True
            note = self._pending
            self._pending = None
            return note

    def infer_s(self) -> float | None:
        with self._lock:
            return self._infer_s

    def _steps_left(self) -> int:
        if self._chunk is None:
            return 0
        return int(len(self._chunk) - self._i)

    def _prefetch_steps(self) -> int:
        n = 1 if self._chunk is None else int(len(self._chunk))
        if self._infer_s is None:
            return max(1, n // 2)
        step_s = SIM_DT * HOLD_TICKS
        need = int(math.ceil(float(self._infer_s) / step_s)) + BLEND_HOLDS
        return max(1, min(need, max(1, n - 1)))

    def wants_frame(self) -> bool:
        with self._lock:
            if not self._ready or self._busy or self._queued is not None:
                return False
            if self._chunk is None:
                return True
            return self._steps_left() <= self._prefetch_steps()

    def wants_still_frame(self) -> bool:
        """True only between chunks. A look taken mid-motion becomes the next yank."""
        with self._lock:
            if not self._ready or self._busy or self._queued is not None:
                return False
            return self._chunk is None

    def submit(self, rgb: np.ndarray, text: str, state: np.ndarray) -> None:
        with self._lock:
            if not self._ready or self._busy or self._queued is not None:
                return
            if self._chunk is not None and self._steps_left() > self._prefetch_steps():
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

    def _install_queued(self) -> None:
        if self._queued is None or self._chunk is not None:
            return
        nxt = self._queued
        self._queued = None
        self._blend_from = None if self._latched is None else self._latched.copy()
        self._blend_left = BLEND_HOLDS * HOLD_TICKS
        self._blend_total = int(self._blend_left)
        self._chunk = nxt
        self._i = 0
        self._hold = 0
        self._anchor = None

    def _blend(self, cmd: np.ndarray) -> np.ndarray:
        if self._blend_left <= 0 or self._blend_from is None or self._blend_total <= 0:
            return cmd
        weight = 1.0 - (float(self._blend_left) / float(self._blend_total))
        mixed = (1.0 - weight) * self._blend_from + weight * cmd
        self._blend_left -= 1
        if self._blend_left <= 0:
            self._blend_from = None
        return clip_command(mixed)

    def command(self, arm_q, lo, hi) -> np.ndarray | None:
        with self._lock:
            self._install_queued()
            if self._chunk is None:
                if self._latched is None:
                    return None
                self._latched = _bleed_base(self._latched)
                return self._latched.copy()
            if self._anchor is None:
                self._anchor = np.asarray(arm_q, dtype=np.float32).reshape(14).copy()
            cmd = self._blend(retarget_action(self._chunk[self._i], self._anchor, lo, hi))
            self._latched = cmd
            self._hold += 1
            if self._hold >= HOLD_TICKS:
                self._hold = 0
                self._i += 1
                self._anchor = None
                if self._i >= len(self._chunk):
                    self._chunk = None
                    self._i = 0
                    self._install_queued()
            return cmd.copy()

    def reset_chunk(self) -> None:
        with self._lock:
            self._gen += 1
            self._chunk = None
            self._i = 0
            self._hold = 0
            self._anchor = None
            self._latched = None
            self._queued = None
            self._blend_from = None
            self._blend_left = 0
            self._blend_total = 0
            self._busy = False
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

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
                            f"System 1.5: инференс {elapsed:.2f} с длиннее чанка {chunk_s:.2f} с."
                        )
                    if gen == int(self._gen):
                        self._queued = chunk
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._pending = f"System 1.5: {self._error}"
            finally:
                with self._lock:
                    self._busy = False

    def _load(self) -> None:
        with TORCH_GATE:
            self._load_weights()

    def _load_weights(self) -> None:
        if not _pi0_enabled():
            raise RuntimeError("PI0_ENABLE=0 — near-field на geometric IK.")
        if not _pi0_tokenizer_available():
            raise RuntimeError(
                "нет HF_TOKEN и локального google/paligemma-3b-pt-224 — "
                "π0 пропущена, near-field на IK."
            )
        policy_cls, make_processors = _import_pi0()
        import os
        import safetensors.torch as safetensors_torch

        # lerobot prints a checkmark after the load. On a cp1251 console that
        # raises, and the caller treats it as a failed load and drops the weights.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        safetensors_torch.load_file = _load_file_nommap
        policy = policy_cls.from_pretrained(PI0_REPO, local_files_only=True)
        policy.eval()
        policy.to(self.device)
        # Processors pull google/paligemma-3b-pt-224 (gated). Need HF_TOKEN + license.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            pre, post = make_processors(
                policy.config,
                PI0_REPO,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        except Exception as exc:
            msg = str(exc)
            gated = any(s in msg.lower() for s in ("401", "gated", "paligemma", "unauthorized", "auth"))
            if gated and not token:
                raise RuntimeError(
                    "π0 processors need HF_TOKEN and access to google/paligemma-3b-pt-224 "
                    "(accept the Gemma license on Hugging Face). Near-field falls back to L2."
                ) from exc
            if gated:
                raise RuntimeError(
                    "π0 processors: gated google/paligemma-3b-pt-224 — accept the license "
                    "on Hugging Face with this HF_TOKEN. Near-field falls back to L2."
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
        obs = {
            _BASE_IMAGE: image,
            _STATE: torch.from_numpy(np.asarray(state, dtype=np.float32)).reshape(-1),
            "task": text or "interact with the object in view",
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
        if arr.shape[-1] < 17:
            pad = np.zeros((arr.shape[0], 17), dtype=np.float32)
            pad[:, : arr.shape[-1]] = arr
            arr = pad
        return arr, time.perf_counter() - t0
