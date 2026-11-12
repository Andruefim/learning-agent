"""Oven flywheel datasets: D_success / D_histories schema (plan §3.3).

Student fields never include privileged geometry. Privileged arrays live in
`privileged.npz` next to the episode for DAgger remapping only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import (
    CHUNK,
    ERROR_LEN,
    INSTR_BYTES,
    STAND_Z,
    TRIAL_MAX,
    VISION_H,
    VISION_STRIDE,
    VISION_W,
    Z_DIM,
)
from agent.h2 import ACTION_DIM, N_ACT, STAND_Q, TRIAL_FEAT
from agent.policy import encode_instr

# Keys that must never appear in student-facing tensors / meta.
PRIVILEGED_KEYS = frozenset(
    {
        "handle_pos",
        "handle_mat",
        "handle_along",
        "hinge_pos",
        "hinge_axis",
        "door_angle",
        "stove_body",
        "handle_geom",
        "qfrc_door",
        "privileged",
    }
)

STUDENT_ARRAY_KEYS = (
    "images",
    "proprio",
    "cmd",
    "language",
    "z",
    "errors",
    "trial_ids",
    "trial_feat",
)


def teacher_cmd_18d(q_des: np.ndarray) -> np.ndarray:
    """Map Tier-A joint targets to the 18-D L3 command space the student emits.

    Base is typically frozen during manipulation → vx/vy/wz = 0; height = STAND_Z;
    arms = q_des[15:29].
    """
    q = np.asarray(q_des, dtype=np.float32).reshape(-1)
    cmd = np.zeros(ACTION_DIM, dtype=np.float32)
    cmd[3] = float(STAND_Z)
    n = min(14, max(0, q.size - 15))
    if n:
        cmd[4 : 4 + n] = q[15 : 15 + n]
    else:
        cmd[4:18] = STAND_Q[15:29]
    return cmd


def proprio_from_qpos(qadr: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    out = np.zeros(N_ACT, dtype=np.float32)
    for i, adr in enumerate(qadr[:N_ACT]):
        out[i] = float(qpos[int(adr)])
    return out


def empty_errors() -> np.ndarray:
    return np.zeros((ERROR_LEN, 3), dtype=np.float32)


def empty_z() -> np.ndarray:
    return np.zeros(Z_DIM, dtype=np.float32)


def empty_trial_ids() -> np.ndarray:
    # Pad token = len(SKILL_IDS) is applied at train time; store zeros here.
    return np.zeros(TRIAL_MAX, dtype=np.int64)


def empty_trial_feat() -> np.ndarray:
    return np.zeros((TRIAL_MAX, TRIAL_FEAT), dtype=np.float32)


class EyeCamera:
    """Lazy MuJoCo head-camera renderer at VISION_H×VISION_W."""

    def __init__(self, model):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.renderer = mujoco.Renderer(model, VISION_H, VISION_W)

    def render(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera="head")
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        self.renderer = None


class EpisodeRecorder:
    """Buffers student-visible ticks (+ optional privileged) at a vision stride."""

    def __init__(
        self,
        *,
        prompt: str,
        seed: int,
        theta: dict,
        attempt: int = 1,
        stride: int | None = None,
    ):
        self.prompt = str(prompt)
        self.seed = int(seed)
        self.theta = {k: float(v) for k, v in dict(theta).items()}
        self.attempt = int(attempt)
        self.stride = max(1, int(stride if stride is not None else VISION_STRIDE))
        self.images: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.cmd: list[np.ndarray] = []
        self.priv_angle: list[float] = []
        self.priv_handle: list[np.ndarray] = []
        self.priv_hinge: list[np.ndarray] = []
        self.tick = 0
        self.language = encode_instr(self.prompt)

    def maybe_record(
        self,
        *,
        image: np.ndarray,
        proprio: np.ndarray,
        cmd: np.ndarray,
        door_angle: float,
        handle_pos: np.ndarray | None = None,
        hinge_pos: np.ndarray | None = None,
    ) -> None:
        if self.tick % self.stride != 0:
            self.tick += 1
            return
        self.images.append(np.asarray(image, dtype=np.uint8).copy())
        self.proprio.append(np.asarray(proprio, dtype=np.float32).reshape(N_ACT).copy())
        self.cmd.append(np.asarray(cmd, dtype=np.float32).reshape(ACTION_DIM).copy())
        self.priv_angle.append(float(door_angle))
        if handle_pos is not None:
            self.priv_handle.append(np.asarray(handle_pos, dtype=np.float32).reshape(3).copy())
        if hinge_pos is not None:
            self.priv_hinge.append(np.asarray(hinge_pos, dtype=np.float32).reshape(3).copy())
        self.tick += 1

    def n_frames(self) -> int:
        return len(self.images)

    def student_payload(self) -> dict[str, Any]:
        t = self.n_frames()
        if t == 0:
            images = np.zeros((0, VISION_H, VISION_W, 3), dtype=np.uint8)
            proprio = np.zeros((0, N_ACT), dtype=np.float32)
            cmd = np.zeros((0, ACTION_DIM), dtype=np.float32)
        else:
            images = np.stack(self.images, axis=0)
            proprio = np.stack(self.proprio, axis=0)
            cmd = np.stack(self.cmd, axis=0)
        return {
            "images": images,
            "proprio": proprio,
            "cmd": cmd,
            "language": self.language.astype(np.int64),
            "z": empty_z(),
            "errors": empty_errors(),
            "trial_ids": empty_trial_ids(),
            "trial_feat": empty_trial_feat(),
        }

    def privileged_payload(self) -> dict[str, Any]:
        t = self.n_frames()
        angles = np.asarray(self.priv_angle, dtype=np.float32)
        handle = (
            np.stack(self.priv_handle, axis=0)
            if self.priv_handle
            else np.zeros((t, 3), dtype=np.float32)
        )
        hinge = (
            np.stack(self.priv_hinge, axis=0)
            if self.priv_hinge
            else np.zeros((t, 3), dtype=np.float32)
        )
        return {
            "door_angle": angles,
            "handle_pos": handle,
            "hinge_pos": hinge,
        }


def episode_dir(root: Path, split: str, seed: int, attempt: int | None = None) -> Path:
    base = Path(root) / split / f"seed_{int(seed):06d}"
    if attempt is None:
        return base
    return base / f"attempt_{int(attempt):02d}"


def save_episode(
    root: Path,
    *,
    split: str,
    seed: int,
    attempt: int,
    recorder: EpisodeRecorder,
    meta: dict[str, Any],
    success: bool,
) -> Path:
    """Write student.npz + privileged.npz + meta.json under D_success or D_histories."""
    if any(k in meta for k in PRIVILEGED_KEYS):
        raise ValueError(f"meta must not contain privileged keys: {PRIVILEGED_KEYS & set(meta)}")
    out = episode_dir(root, split, seed, attempt)
    out.mkdir(parents=True, exist_ok=True)
    student = recorder.student_payload()
    assert_student_clean(student)
    np.savez_compressed(out / "student.npz", **student)
    np.savez_compressed(out / "privileged.npz", **recorder.privileged_payload())
    payload = {
        "seed": int(seed),
        "attempt": int(attempt),
        "prompt": recorder.prompt,
        "theta": recorder.theta,
        "success": bool(success),
        "n_frames": int(recorder.n_frames()),
        "vision": {"h": VISION_H, "w": VISION_W, "stride": int(getattr(recorder, "stride", VISION_STRIDE))},
        "action_dim": ACTION_DIM,
        "chunk": CHUNK,
        **meta,
    }
    (out / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_history_index(root: Path, seed: int, attempts: list[dict[str, Any]]) -> Path:
    out = episode_dir(root, "D_histories", seed)
    out.mkdir(parents=True, exist_ok=True)
    for a in attempts:
        if PRIVILEGED_KEYS & set(a):
            raise ValueError("history attempt leaked privileged keys")
    payload = {"seed": int(seed), "attempts": attempts}
    path = out / "history.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def assert_student_clean(obj: Any, *, path: str = "root") -> None:
    """Fail if privileged field names appear anywhere in student-facing structures."""
    if isinstance(obj, dict):
        bad = PRIVILEGED_KEYS & set(obj)
        if bad:
            raise AssertionError(f"privileged leak at {path}: {sorted(bad)}")
        for k, v in obj.items():
            assert_student_clean(v, path=f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_student_clean(v, path=f"{path}[{i}]")


def validate_episode_dir(path: Path) -> list[str]:
    """Return a list of schema errors (empty = ok)."""
    errs: list[str] = []
    path = Path(path)
    meta_p = path / "meta.json"
    stud_p = path / "student.npz"
    priv_p = path / "privileged.npz"
    if not meta_p.exists():
        errs.append("missing meta.json")
        return errs
    if not stud_p.exists():
        errs.append("missing student.npz")
    if not priv_p.exists():
        errs.append("missing privileged.npz")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"bad meta.json: {e}"]
    try:
        assert_student_clean(meta)
    except AssertionError as e:
        errs.append(str(e))
    if not stud_p.exists():
        return errs
    with np.load(stud_p) as z:
        for key in STUDENT_ARRAY_KEYS:
            if key not in z.files:
                errs.append(f"student.npz missing {key}")
        if "images" in z.files:
            img = z["images"]
            if img.ndim != 4 or img.shape[-1] != 3:
                errs.append(f"images shape {img.shape}, want (T,H,W,3)")
            elif img.shape[1:3] != (VISION_H, VISION_W):
                errs.append(f"images HxW {img.shape[1:3]} != {(VISION_H, VISION_W)}")
        if "proprio" in z.files and z["proprio"].ndim == 2 and z["proprio"].shape[-1] != N_ACT:
            errs.append(f"proprio dim {z['proprio'].shape[-1]} != {N_ACT}")
        if "cmd" in z.files and z["cmd"].ndim == 2 and z["cmd"].shape[-1] != ACTION_DIM:
            errs.append(f"cmd dim {z['cmd'].shape[-1]} != {ACTION_DIM}")
        if "language" in z.files and z["language"].shape != (INSTR_BYTES,):
            errs.append(f"language shape {z['language'].shape} != ({INSTR_BYTES},)")
        try:
            assert_student_clean({k: True for k in z.files})
        except AssertionError as e:
            errs.append(str(e))
    if priv_p.exists():
        with np.load(priv_p) as z:
            if "door_angle" not in z.files:
                errs.append("privileged.npz missing door_angle")
    if "success" not in meta:
        errs.append("meta missing success")
    return errs


def validate_dataset_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    report = {
        "root": str(root),
        "D_success": 0,
        "D_histories": 0,
        "D_histories_with_recovery": 0,
        "errors": [],
    }
    success_root = root / "D_success"
    if success_root.exists():
        for meta in success_root.rglob("meta.json"):
            report["D_success"] += 1
            errs = validate_episode_dir(meta.parent)
            for e in errs:
                report["errors"].append(f"{meta.parent}: {e}")
    hist_root = root / "D_histories"
    if hist_root.exists():
        for hist in hist_root.glob("seed_*/history.json"):
            report["D_histories"] += 1
            try:
                data = json.loads(hist.read_text(encoding="utf-8"))
                attempts = data.get("attempts") or []
                assert_student_clean(data)
                saw_fail = False
                recovered = False
                for a in attempts:
                    if not a.get("success"):
                        saw_fail = True
                    elif saw_fail:
                        recovered = True
                if recovered:
                    report["D_histories_with_recovery"] += 1
            except Exception as e:
                report["errors"].append(f"{hist}: {e}")
            for meta in hist.parent.rglob("meta.json"):
                for e in validate_episode_dir(meta.parent):
                    report["errors"].append(f"{meta.parent}: {e}")
    report["ok"] = len(report["errors"]) == 0
    return report


def chunkify_cmd(cmd: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """(T, A) → (N, chunk, A) with zero pad on the last window."""
    cmd = np.asarray(cmd, dtype=np.float32)
    if cmd.ndim != 2:
        raise ValueError("cmd must be (T, A)")
    t, a = cmd.shape
    if t == 0:
        return np.zeros((0, chunk, a), dtype=np.float32)
    pad = (-t) % chunk
    if pad:
        cmd = np.concatenate([cmd, np.zeros((pad, a), dtype=np.float32)], axis=0)
    return cmd.reshape(-1, chunk, a)
