"""Real Pretrained E2E Embodied Backbone: lerobot/pi0_base (bfloat16) + Smooth G1 Action Adapter."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
import types
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from agent.config import SLEEP_ACTION_DIM, SLEEP_HORIZON
from agent.system0 import ACT_HI, ACT_LO, raw_to_norm_action

PI0_DEFAULT_REPO = "lerobot/pi0_base"
_BASE_IMAGE = "observation.images.base_0_rgb"
_STATE_KEY = "observation.state"

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


def pi0_enabled() -> bool:
    raw = os.getenv("E2E_ENABLE", os.getenv("PI0_ENABLE", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _torch_distributed_ok() -> bool:
    try:
        import torch.distributed.distributed_c10d  # noqa: F401
        return True
    except Exception:
        return False


def _import_pi0():
    """Import PI0Policy without pulling torchao when torch.distributed is missing."""
    if not _torch_distributed_ok():
        import transformers.utils as tu
        import transformers.utils.import_utils as iu

        iu.is_torchao_available = lambda: False
        tu.is_torchao_available = lambda: False
        import lerobot

        if "lerobot.policies" not in sys.modules:
            policies = types.ModuleType("lerobot.policies")
            policies.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
            policies.__package__ = "lerobot.policies"
            sys.modules["lerobot.policies"] = policies
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    return PreTrainedConfig, PI0Policy, make_pre_post_processors


def resolve_pi0_source(repo_or_path: str = PI0_DEFAULT_REPO) -> Tuple[str, bool]:
    """Return (snapshot_path_or_repo, local_files_only)."""
    candidate = Path(repo_or_path)
    if candidate.is_dir() and (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
        return str(candidate), True

    # Prefer complete local HuggingFace snapshot if present (handles incomplete second snapshots cleanly)
    if repo_or_path.startswith("lerobot/pi0"):
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--lerobot--pi0_base" / "snapshots"
        if hub_dir.is_dir():
            complete = [
                p
                for p in hub_dir.iterdir()
                if p.is_dir()
                and (p / "config.json").is_file()
                and (p / "model.safetensors").is_file()
                and (p / "policy_preprocessor.json").is_file()
            ]
            if complete:
                best = max(complete, key=lambda p: (p / "config.json").stat().st_mtime)
                return str(best), True
    return repo_or_path, False


def read_safetensors_bf16(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read a safetensors checkpoint without mmap (avoids PyTorch Windows ROCm faults)."""
    p = Path(path)
    out: dict[str, torch.Tensor] = {}
    with p.open("rb") as h:
        header_n = struct.unpack("<Q", h.read(8))[0]
        header = json.loads(h.read(header_n))
        base = 8 + header_n
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            h.seek(base + int(start))
            buf = bytearray(h.read(int(end) - int(start)))
            t = torch.frombuffer(buf, dtype=_SAFETENSORS_DTYPE[spec["dtype"]]).reshape(spec["shape"])
            t = t.to(dtype=torch.bfloat16) if t.is_floating_point() else t.clone()
            out[key] = t if str(device) == "cpu" else t.to(device)
    return out


def build_pi0_state(proprio_28: np.ndarray, cur_raw: np.ndarray, wrist_wrench: float) -> np.ndarray:
    """Pack G1 28-D proprioception + active motor target + tactile wrist load into 32-D state."""
    st = np.zeros(32, dtype=np.float32)
    pr = np.asarray(proprio_28, dtype=np.float32).reshape(-1)
    n = min(28, int(pr.shape[0]))
    st[:n] = pr[:n]
    raw = np.asarray(cur_raw, dtype=np.float32).reshape(SLEEP_ACTION_DIM)
    st[28] = float(raw[2])   # squat
    st[29] = float(raw[4])   # waist_pitch
    st[30] = float(raw[19])  # right grip
    st[31] = float(np.tanh(wrist_wrench / 15.0))
    return st


def pi0_chunk_to_g1_norm(raw_chunk: np.ndarray, cur_raw: np.ndarray) -> np.ndarray:
    """Retarget a pi0 (T, >=20) action chunk into smoothed (SLEEP_HORIZON, 20) G1 [-1, 1] space."""
    src = np.asarray(raw_chunk, dtype=np.float32)
    if src.ndim == 3:
        src = src[0]
    src = src.reshape(-1, src.shape[-1])
    T = max(1, int(src.shape[0]))

    # Subsample evenly across the responsive front horizon of pi0's chunk (~first 18 steps)
    span = min(T, 18)
    idxs = np.linspace(0, max(0, span - 1), SLEEP_HORIZON).round().astype(int)
    base_raw = np.asarray(cur_raw, dtype=np.float32).reshape(SLEEP_ACTION_DIM).copy()
    arm_delta_scale = 0.14
    waist_delta_scale = 0.08

    out = np.zeros((SLEEP_HORIZON, SLEEP_ACTION_DIM), dtype=np.float32)
    running = base_raw.copy()
    for step_i, idx in enumerate(idxs):
        vec = np.zeros(32, dtype=np.float32)
        m = min(32, int(src.shape[-1]))
        vec[:m] = np.clip(src[idx, :m], -1.0, 1.0)

        nxt = running.copy()
        # Arms smoothly follow pi0 action expert deltas anchored at running posture
        nxt[5:12] = np.clip(running[5:12] + vec[0:7] * arm_delta_scale, ACT_LO[5:12], ACT_HI[5:12])
        nxt[12:19] = np.clip(running[12:19] + vec[7:14] * arm_delta_scale, ACT_LO[12:19], ACT_HI[12:19])
        # Gripper channel + gentle waist/squat modulation
        nxt[19] = float(np.clip(0.5 * (vec[15] + 1.0) if abs(vec[15]) > 0.05 else running[19], 0.0, 1.0))
        nxt[2] = float(np.clip(running[2] + vec[17] * 0.04, 0.0, 0.40))
        nxt[3] = float(np.clip(running[3] + vec[18] * waist_delta_scale, -0.45, 0.45))
        nxt[4] = float(np.clip(running[4] + vec[19] * waist_delta_scale, -0.15, 0.25))
        # Subtle closed-loop heading/step micro-adjustment with deadzone for balance safety
        vx = float(vec[14]) * 0.35 if abs(float(vec[14])) > 0.18 else 0.0
        wz = float(vec[16]) * 0.35 if abs(float(vec[16])) > 0.18 else 0.0
        nxt[0] = float(np.clip(vx, -0.25, 0.35))
        nxt[1] = float(np.clip(wz, -0.40, 0.40))

        running = nxt
        out[step_i] = raw_to_norm_action(nxt)
    return out


class Pi0E2EModel:
    """Async-loaded lerobot/pi0_base policy synchronized on the shared PyTorch lock."""

    def __init__(self, device: torch.device, net_lock: threading.RLock, *, log=print) -> None:
        self.device = device
        self._net_lock = net_lock
        self._log = log
        self._lock = threading.Lock()
        self._policy = None
        self._pre = None
        self._post = None
        self._phase = "off"   # off | loading | ready | error
        self._error = ""
        self._thread: Optional[threading.Thread] = None
        self.last_infer_ms = 0.0

    def phase(self) -> str:
        with self._lock:
            return self._phase

    def is_ready(self) -> bool:
        with self._lock:
            return self._phase == "ready" and self._policy is not None

    def error(self) -> str:
        with self._lock:
            return self._error

    def start_async(self) -> None:
        if not pi0_enabled() or self.device.type != "cuda":
            with self._lock:
                self._phase = "off"
            return
        with self._lock:
            if self._phase in ("loading", "ready") or self._thread is not None:
                return
            self._phase = "loading"
        self._thread = threading.Thread(target=self._load_worker, name="pi0-e2e-loader", daemon=True)
        self._thread.start()

    def _load_worker(self) -> None:
        try:
            t0 = time.perf_counter()
            repo_or_snap, local_only = resolve_pi0_source(os.getenv("E2E_REPO", PI0_DEFAULT_REPO).strip() or PI0_DEFAULT_REPO)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            PreTrainedConfig, PI0Policy, make_pre_post_processors = _import_pi0()
            import safetensors.torch as st

            st.load_file = read_safetensors_bf16
            cfg = PreTrainedConfig.from_pretrained(repo_or_snap, local_files_only=local_only)
            cfg.dtype = "bfloat16"
            policy = PI0Policy.from_pretrained(repo_or_snap, config=cfg, local_files_only=local_only)

            pre, post = make_pre_post_processors(
                policy.config,
                repo_or_snap,
                local_files_only=local_only,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            with self._net_lock:
                policy.eval().to(device=self.device, dtype=torch.bfloat16)

            with self._lock:
                self._policy = policy
                self._pre = pre
                self._post = post
                self._phase = "ready"
                self._error = ""
            vram_gb = torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
            self._log(f"[E2E π₀] Готова за {time.perf_counter() - t0:.1f}с (bfloat16, VRAM={vram_gb:.2f} ГБ).")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:180]
            with self._lock:
                self._phase = "error"
                self._error = err
            self._log(f"[E2E π₀] Не удалось загрузить π₀ ({err}) — работаю через CMS System 1.")

    def propose_chunk(
        self,
        head_rgb: np.ndarray,
        proprio_28: np.ndarray,
        goal_en: str,
        cur_raw: np.ndarray,
        wrist_wrench: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Generate a smoothed (SLEEP_HORIZON, 20) normalized G1 action chunk from pi0_base."""
        with self._lock:
            if self._phase != "ready" or self._policy is None or self._pre is None:
                return None
            policy, pre, post = self._policy, self._pre, self._post

        t0 = time.perf_counter()
        image = torch.from_numpy(np.ascontiguousarray(head_rgb)).permute(2, 0, 1).float().div_(255.0)
        state = torch.from_numpy(build_pi0_state(proprio_28, cur_raw, wrist_wrench))
        obs = {
            _BASE_IMAGE: image,
            _STATE_KEY: state,
            "task": (goal_en or "interact with the kitchen appliance in front").strip(),
        }
        with self._net_lock:
            batch = pre(obs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                raw_chunk = policy.predict_action_chunk(batch)
            try:
                raw_chunk = post(raw_chunk)
            except Exception:
                pass
            arr = raw_chunk.detach().float().cpu().numpy()
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return pi0_chunk_to_g1_norm(arr, cur_raw)
