"""Tier B theta loop unit checks (no MuJoCo episode)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.oven_task import THETA_BOUNDS, clamp_theta  # noqa: E402
from agent.oven_tier_b import (  # noqa: E402
    TIER_B_START_THETA,
    AttemptRecord,
    perturb_theta,
    random_theta,
    validate_theta,
)


def test_start_theta_suboptimal_vs_defaults():
    # Headroom for K>1: stance farther / slower pull than production defaults.
    assert TIER_B_START_THETA["stance_x"] <= -0.5
    assert TIER_B_START_THETA["pull_speed"] <= 0.2


def test_random_and_perturb_in_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        th = random_theta(rng)
        for k, (lo, hi) in THETA_BOUNDS.items():
            assert lo <= th[k] <= hi
        th2 = perturb_theta(rng, th, scale=0.5)
        for k, (lo, hi) in THETA_BOUNDS.items():
            assert lo <= th2[k] <= hi


def test_attempt_record_text_has_fields():
    rec = AttemptRecord(
        k=1,
        theta=clamp_theta(TIER_B_START_THETA),
        max_angle=0.2,
        success=False,
        fell=False,
        contact=True,
        slip=False,
        min_hand_dist=0.05,
        outcome="timeout",
        reason="timeout",
    )
    txt = rec.as_text()
    assert "contact=True" in txt
    assert "min_hand_dist=" in txt
    assert validate_theta(rec.theta) is not None
