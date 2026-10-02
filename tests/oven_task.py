"""Oven task metric + randomization + zero-policy gate helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"

from agent.oven_task import (  # noqa: E402
    OvenScene,
    is_success,
    oven_open_angle,
    privileged_handle,
    run_zero_policy_episode,
)


def test_oven_angle_closed_at_reset():
    scene = OvenScene()
    scene.reset(0, randomize=False)
    assert oven_open_angle(scene.model, scene.data) < 1e-4


def test_is_success_hold():
    dt = 0.002
    closed = [0.0] * 100
    assert not is_success(closed, dt=dt)
    open_short = [0.9] * int(0.5 / dt)
    assert not is_success(open_short, dt=dt)
    open_ok = [0.9] * int(1.0 / dt + 5)
    assert is_success(open_ok, dt=dt)


def test_randomize_reproducible():
    a = OvenScene()
    b = OvenScene()
    ra = a.reset(7, randomize=True)
    rb = b.reset(7, randomize=True)
    assert ra["spawn_dx"] == rb["spawn_dx"]
    assert ra["prompt"] == rb["prompt"]
    assert np.allclose(a.data.qpos[:7], b.data.qpos[:7])
    assert oven_open_angle(a.model, a.data) == oven_open_angle(b.model, b.data)


def test_privileged_handle_fields():
    scene = OvenScene()
    scene.reset(1, randomize=False)
    info = privileged_handle(scene.model, scene.data)
    assert info["handle_pos"].shape == (3,)
    assert abs(float(np.linalg.norm(info["hinge_axis"])) - 1.0) < 1e-5
    assert info["door_angle"] < 1e-4


def test_zero_policy_no_open():
    r = run_zero_policy_episode(0, seconds=2.0, randomize=True)
    assert not r.success
    assert not r.fell
    assert r.max_angle < 0.2
