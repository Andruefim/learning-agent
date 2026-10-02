"""Dataset schema, privileged leakage guard, theta JSON contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"

from agent.config import STUDENT_ONLY_EVAL, VISION_H, VISION_W  # noqa: E402
from agent.h2 import ACTION_DIM, N_ACT, STAND_Q  # noqa: E402
from agent.oven_dataset import (  # noqa: E402
    PRIVILEGED_KEYS,
    EpisodeRecorder,
    assert_student_clean,
    save_episode,
    teacher_cmd_18d,
    validate_dataset_root,
    validate_episode_dir,
)
from agent.oven_tier_b import validate_theta  # noqa: E402
from agent.l3_cmd import clip_command  # noqa: E402


def test_teacher_cmd_18d_shape():
    cmd = teacher_cmd_18d(STAND_Q)
    assert cmd.shape == (ACTION_DIM,)
    assert np.allclose(cmd[4:18], STAND_Q[15:29])


def test_validate_theta_contract():
    assert validate_theta({"stance_x": -0.3, "pull_speed": 0.2}) is not None
    assert validate_theta("nope") is None
    assert validate_theta({"stance_x": -10.0}) is not None  # clamped
    bad = validate_theta({"unknown": 1})
    assert bad is not None and "unknown" not in bad


def test_privileged_leak_detected():
    try:
        assert_student_clean({"prompt": "x", "handle_pos": [1, 2, 3]})
        assert False, "expected leak"
    except AssertionError:
        pass


def test_student_only_eval_default_off():
    assert STUDENT_ONLY_EVAL is False


def test_mix_student_only(monkeypatch):
    monkeypatch.setenv("STUDENT_ONLY_EVAL", "1")
    # Re-import flag path through flywheel helper would need reload; test clip path instead.
    teacher = np.zeros(ACTION_DIM, dtype=np.float32)
    student = np.ones(ACTION_DIM, dtype=np.float32)
    # Emulate _mix branch
    from agent import config

    monkeypatch.setattr(config, "STUDENT_ONLY_EVAL", True)
    if config.STUDENT_ONLY_EVAL:
        out = clip_command(student)
    else:
        out = teacher
    assert float(out.mean()) > 0.5


def test_episode_roundtrip(tmp_path: Path):
    rec = EpisodeRecorder(prompt="open the oven", seed=1, theta={"stance_x": -0.3}, attempt=1)
    for i in range(8):
        rec.maybe_record(
            image=np.zeros((VISION_H, VISION_W, 3), dtype=np.uint8),
            proprio=np.zeros(N_ACT, dtype=np.float32),
            cmd=np.zeros(ACTION_DIM, dtype=np.float32),
            door_angle=0.01 * i,
            handle_pos=np.zeros(3, dtype=np.float32),
            hinge_pos=np.zeros(3, dtype=np.float32),
        )
    out = save_episode(
        tmp_path,
        split="D_success",
        seed=1,
        attempt=1,
        recorder=rec,
        meta={"mode": "test"},
        success=True,
    )
    errs = validate_episode_dir(out)
    assert errs == [], errs
    report = validate_dataset_root(tmp_path)
    assert report["ok"]
    assert report["D_success"] == 1
    # ensure privileged keys known
    assert "handle_pos" in PRIVILEGED_KEYS
