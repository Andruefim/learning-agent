"""Tier B: external in-context θ loop over Tier A skills.

The LLM never outputs joints and never judges success — only proposes θ JSON.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from agent.oven_task import (
    DEFAULT_THETA,
    THETA_BOUNDS,
    OvenScene,
    clamp_theta,
)
from agent.oven_teacher import run_tier_a_episode

# Suboptimal start so K>1 methods have headroom (production Tier A defaults ~100%).
TIER_B_START_THETA = {
    **DEFAULT_THETA,
    "stance_x": -0.55,
    "pregrasp_clearance": 0.14,
    "pull_speed": 0.15,
    "contact_offset_along_handle": 0.22,
    "pull_arc_scale": 0.75,
}


@dataclass
class AttemptRecord:
    k: int
    theta: dict
    max_angle: float
    success: bool
    fell: bool
    contact: bool
    slip: bool
    min_hand_dist: float
    outcome: str
    reason: str

    def as_text(self) -> str:
        return (
            f"attempt {self.k}: success={self.success} fell={self.fell} "
            f"max_angle={self.max_angle:.3f} contact={self.contact} slip={self.slip} "
            f"min_hand_dist={self.min_hand_dist:.3f} outcome={self.outcome} "
            f"theta={json.dumps({k: round(float(v), 3) for k, v in self.theta.items()})}"
        )


@dataclass
class SceneResult:
    seed: int
    mode: str
    success: bool
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_theta: dict = field(default_factory=dict)
    llm_log: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "mode": self.mode,
            "success": self.success,
            "final_theta": self.final_theta,
            "attempts": [asdict(a) for a in self.attempts],
            "llm_log": self.llm_log,
        }


def validate_theta(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    try:
        return clamp_theta(raw)
    except Exception:
        return None


def random_theta(rng: np.random.Generator, base: dict | None = None) -> dict:
    out = dict(base or TIER_B_START_THETA)
    for k, (lo, hi) in THETA_BOUNDS.items():
        out[k] = float(rng.uniform(lo, hi))
    return clamp_theta(out)


def perturb_theta(rng: np.random.Generator, theta: dict, scale: float = 0.15) -> dict:
    out = dict(theta)
    for k, (lo, hi) in THETA_BOUNDS.items():
        span = hi - lo
        out[k] = float(out.get(k, DEFAULT_THETA[k])) + float(rng.normal(0.0, scale * span))
    return clamp_theta(out)


class ThetaLLM:
    """Ollama JSON proposer. Falls back to perturb on failure."""

    def __init__(self):
        self.model = os.getenv("L1_MODEL", "qwen3.5:9b-q8_0").strip()
        from agent.planner import resolve_ollama_url

        self.base_url = resolve_ollama_url()
        self.last_err = ""

    def propose(self, history: list[AttemptRecord], prompt: str) -> tuple[dict, dict]:
        bounds = {k: list(v) for k, v in THETA_BOUNDS.items()}
        hist_txt = "\n".join(a.as_text() for a in history) or "(none)"
        user = (
            "You tune skill parameters θ for a robot opening an oven door.\n"
            f"Command: {prompt}\n"
            f"Allowed keys and bounds: {json.dumps(bounds)}\n"
            f"Previous attempts:\n{hist_txt}\n"
            "Propose the next θ as a single JSON object with only those keys (numbers).\n"
            "Do not output joints. Do not judge success. Improve based on the history."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 256},
        }
        log = {"request": user, "response": "", "ok": False}
        try:
            import httpx

            with httpx.Client(timeout=45.0) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                text = r.json()["message"]["content"]
            log["response"] = text
            text = text.strip()
            if "</think>" in text:
                text = text.split("</think>", 1)[-1]
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1] if start >= 0 else text)
            theta = validate_theta(data)
            if theta is None:
                self.last_err = "invalid theta"
                return clamp_theta(history[-1].theta if history else TIER_B_START_THETA), log
            log["ok"] = True
            self.last_err = ""
            return theta, log
        except Exception as e:
            self.last_err = f"{type(e).__name__}: {e}"[:160]
            log["response"] = self.last_err
            base = history[-1].theta if history else TIER_B_START_THETA
            return perturb_theta(np.random.default_rng(), base), log


Mode = Literal["k1_default", "k5_random", "k5_llm"]


def run_scene(
    seed: int,
    mode: Mode,
    *,
    k_max: int = 5,
    seconds: float = 12.0,
    scene: OvenScene | None = None,
    llm: ThetaLLM | None = None,
) -> SceneResult:
    """One randomized scene, up to k_max attempts under the chosen θ update rule."""
    own = scene is None
    scene = scene or OvenScene()
    rng = np.random.default_rng(int(seed) + 17)
    # Freeze scene randomization across attempts of the same seed.
    applied = scene.reset(int(seed), randomize=True, theta=TIER_B_START_THETA)
    prompt = str(applied.get("prompt") or scene.prompt)
    # Preserve randomized model state by snapshotting qpos/model knobs via seed only —
    # each attempt re-resets with the same seed so randomization matches.
    history: list[AttemptRecord] = []
    llm_log: list[dict] = []
    theta = clamp_theta(TIER_B_START_THETA)
    k_limit = 1 if mode == "k1_default" else int(k_max)

    for k in range(1, k_limit + 1):
        if k > 1:
            if mode == "k5_random":
                theta = random_theta(rng, base=TIER_B_START_THETA)
            elif mode == "k5_llm":
                llm = llm or ThetaLLM()
                theta, log = llm.propose(history, prompt)
                llm_log.append(log)
            else:
                theta = clamp_theta(TIER_B_START_THETA)
        row = run_tier_a_episode(
            int(seed),
            seconds=seconds,
            randomize=True,
            theta=theta,
            scene=scene,
        )
        rec = AttemptRecord(
            k=k,
            theta=dict(theta),
            max_angle=float(row["max_angle"]),
            success=bool(row["success"]),
            fell=bool(row["fell"]),
            contact=bool(row["extras"].get("contact")),
            slip=bool(row["extras"].get("slip")),
            min_hand_dist=float(row["extras"].get("min_hand_dist") or 1.0),
            outcome=str(row["outcome"]),
            reason=str(row["reason"]),
        )
        history.append(rec)
        if rec.success:
            break

    result = SceneResult(
        seed=int(seed),
        mode=mode,
        success=any(a.success for a in history),
        attempts=history,
        final_theta=dict(history[-1].theta if history else theta),
        llm_log=llm_log,
    )
    if own:
        pass
    return result
