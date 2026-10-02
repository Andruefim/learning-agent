"""Whole-body e2e teacher contract."""

from __future__ import annotations

import numpy as np
import torch

from agent.e2e import E2ETeacher, e2e_enabled, resolve_e2e_device, retarget_body
from agent.h2 import N_ACT, STAND_Q


def test_retarget_body_stays_in_limits():
    lo = np.full(N_ACT, -2.0, np.float32)
    hi = np.full(N_ACT, 2.0, np.float32)
    q = STAND_Q.copy()
    action = np.ones(32, np.float32)
    out = retarget_body(action, q, lo, hi)
    assert out.shape == (N_ACT,)
    assert np.all(out <= hi + 1e-5) and np.all(out >= lo - 1e-5)
    assert float(np.max(np.abs(out - q))) > 0.01


def test_retarget_body_pads_short_actions():
    lo = np.full(N_ACT, -3.0, np.float32)
    hi = np.full(N_ACT, 3.0, np.float32)
    q = np.zeros(N_ACT, np.float32)
    out = retarget_body(np.array([1.0, -1.0], np.float32), q, lo, hi)
    assert out.shape == (N_ACT,)
    assert abs(float(out[0])) > 0.0
    assert float(out[2]) == 0.0


def test_load_now_refuses_cpu():
    teacher = E2ETeacher(torch.device("cpu"))
    note = teacher.load_now()
    assert note is not None and "CPU" in note
    assert teacher.error
    assert teacher._ready is False


def test_e2e_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("E2E_ENABLE", "1")
    assert e2e_enabled() is True
    monkeypatch.setenv("E2E_ENABLE", "0")
    assert e2e_enabled() is False


def test_resolve_e2e_device_cpu_explicit(monkeypatch):
    monkeypatch.setenv("E2E_DEVICE", "cpu")
    assert resolve_e2e_device().type == "cpu"


if __name__ == "__main__":
    test_retarget_body_stays_in_limits()
    test_retarget_body_pads_short_actions()
    test_load_now_refuses_cpu()
    print("ok")
