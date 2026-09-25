"""System 1.5. π0 proposes a chunk; the walk net still owns the legs.

The checkpoint is lerobot/pi0_base (Gemma license). It is loaded on the first
interaction that names a pixel, on a side thread, so physics does not wait on
the download. Wrist cameras are not added: missing views stay masked.

Its 32-D normalized action is retargeted into the existing 18-D command.
Dims 0:14 are arm deltas in actuator order. Dims 14 and 16 are forward and
yaw speed. Lateral speed stays 0 because this gait falls when it strafes.
Height stays at the stand. Fingers are not in the chunk.
"""

from __future__ import annotations

import queue
import sys
import threading
import types
from pathlib import Path

import numpy as np
import torch

from agent.config import STAND_Z
from agent.h2 import colliding_geoms
from agent.l3_cmd import CMD_ARMS, CMD_H, CMD_VX, CMD_VY, CMD_WZ, clip_command, stand_command

PI0_REPO = "lerobot/pi0_base"
VLA_H, VLA_W = 240, 320
ARM_DELTA = 0.12
BASE_VX = 0.25
BASE_WZ = 0.40
BASE_DEAD = 0.15
HOLD_TICKS = 10
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


def _torch_distributed_ok() -> bool:
    try:
        import torch.distributed.distributed_c10d  # noqa: F401
    except Exception:
        return False
    return True


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

    def ensure_started(self) -> str | None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="pi0", daemon=True)
                self._thread.start()
                self._pending = "System 1.5: гружу lerobot/pi0_base."
            note = self._pending
            self._pending = None
            return note

    def wants_frame(self) -> bool:
        with self._lock:
            return bool(self._ready and not self._busy and self._chunk is None)

    def submit(self, rgb: np.ndarray, text: str, state: np.ndarray) -> None:
        with self._lock:
            if not self._ready or self._busy or self._chunk is not None:
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

    def command(self, arm_q, lo, hi) -> np.ndarray | None:
        with self._lock:
            if self._chunk is None:
                return None if self._latched is None else self._latched.copy()
            if self._anchor is None:
                self._anchor = np.asarray(arm_q, dtype=np.float32).reshape(14).copy()
            action = self._chunk[self._i]
            cmd = retarget_action(action, self._anchor, lo, hi)
            self._latched = cmd
            self._hold += 1
            if self._hold >= HOLD_TICKS:
                self._hold = 0
                self._i += 1
                self._anchor = None
                if self._i >= len(self._chunk):
                    self._chunk = None
                    self._i = 0
            return cmd.copy()

    def reset_chunk(self) -> None:
        with self._lock:
            self._gen += 1
            self._chunk = None
            self._i = 0
            self._hold = 0
            self._anchor = None
            self._latched = None
            self._busy = False
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

    def _loop(self) -> None:
        try:
            self._load()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._pending = f"System 1.5 не поднялась: {self._error}"
            return
        with self._lock:
            self._ready = True
            self._pending = "System 1.5 готова."
        while True:
            rgb, text, state, gen = self._jobs.get()
            try:
                with self._lock:
                    current = int(self._gen)
                if gen != current:
                    continue
                chunk = self._infer(rgb, text, state)
                with self._lock:
                    if gen == int(self._gen):
                        self._chunk = chunk
                        self._i = 0
                        self._hold = 0
                        self._anchor = None
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._pending = f"System 1.5: {self._error}"
            finally:
                with self._lock:
                    self._busy = False

    def _load(self) -> None:
        policy_cls, make_processors = _import_pi0()
        policy = policy_cls.from_pretrained(PI0_REPO)
        policy.eval()
        policy.to(self.device)
        pre, post = make_processors(
            policy.config,
            PI0_REPO,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self._policy = policy
        self._pre = pre
        self._post = post

    def _infer(self, rgb: np.ndarray, text: str, state: np.ndarray) -> np.ndarray:
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
        return arr
