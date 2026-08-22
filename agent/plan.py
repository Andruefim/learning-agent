"""L1 skill/params → teacher prior and trial scoring. Not the motor contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from agent.config import (
    DEFAULT_PARAMS,
    PARAM_KEYS,
    PARAM_OK,
    PARAM_SCALE,
    SKILL_TO_I,
    STAND_Z,
    Z_DIM,
)


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class TeacherIntent:
    """Numeric prior used only as tracker teacher / distillation label. Not the L1 contract."""

    height: float = 1.0
    vx: float = 0.0
    yaw: float = 0.0
    r_arm: float = 0.0
    l_arm: float = 0.0
    r_out: float = 0.0
    l_out: float = 0.0
    steps: int = 0
    wave: float = 0.0
    kick: float = 0.0


def _soft(params: dict, key: str, default: str = "") -> str:
    if not params:
        return default
    val = params.get(key, default)
    return str(val).strip().lower()


def decode_teacher(skill: str, params: dict | None) -> TeacherIntent:
    skill = (skill or "hold").strip().lower()
    p = params or {}
    t = TeacherIntent()
    if skill == "squat":
        t.height = {"low": 0.45, "deep": 0.38, "medium": 0.62, "high": 0.78}.get(_soft(p, "depth", "low"), 0.45)
    if skill == "locomote":
        speed = {"slow": 0.30, "medium": 0.50, "fast": 0.80}.get(_soft(p, "speed", "medium"), 0.50)
        direction = _soft(p, "direction", "forward")
        t.vx = -speed if direction in {"back", "backward", "назад"} else speed
        hint = _soft(p, "distance_hint") or str(p.get("steps", ""))
        found = re.findall(r"\d+", hint)
        t.steps = int(found[0]) if found else 0
    if skill == "turn":
        direction = _soft(p, "direction", "left")
        t.yaw = -1.0 if direction in {"right", "направо", "cw"} else 1.0
    if skill in {"wave", "reach"}:
        hand = _soft(p, "hand", "right" if skill == "wave" else "both")
        pose = _soft(p, "pose")
        if hand in {"right", "both", "правой", "правая"}:
            t.r_arm = 1.0 if skill == "wave" else 0.85
        if hand in {"left", "both", "левой", "левая"}:
            t.l_arm = 1.0 if skill == "wave" else 0.85
        if skill == "wave":
            t.wave = 1.0
        if pose in {"t", "out", "sides"}:
            t.r_out, t.l_out, t.r_arm, t.l_arm, t.wave = 1.0, 1.0, 0.35, 0.35, 0.0
        if pose == "clap":
            t.r_arm, t.l_arm, t.r_out, t.l_out, t.wave = 0.55, 0.55, -0.7, -0.7, 1.0
        if skill == "reach" and _soft(p, "hands") == "down":
            t.r_arm = t.l_arm = t.wave = 0.0
    if skill == "hold" and _soft(p, "hands") == "down":
        t.r_arm = t.l_arm = t.wave = 0.0
    if skill == "kick":
        foot = _soft(p, "foot", "right")
        t.kick = -1.0 if foot in {"left", "левой"} else 1.0
    return t


def parse_requested_yaw(skill: str, params: dict | None) -> float | None:
    """World-frame yaw goal in radians, or None if L1 did not name an angle."""
    if (skill or "").strip().lower() != "turn":
        return None
    p = params or {}
    direction = _soft(p, "direction", "left")
    sign = -1.0 if direction in {"right", "направо", "cw"} else 1.0
    raw = p.get("angle", p.get("requested_yaw", p.get("degrees")))
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        found = re.findall(r"-?\d+\.?\d*", raw)
        if not found:
            return None
        val = float(found[0])
    else:
        val = float(raw)
    if abs(val) > np.pi + 0.05:
        val = float(np.deg2rad(val))
    return sign * abs(val)


@dataclass
class Plan:
    instruction: str = "stand"
    skill: str = "hold"
    params: dict = field(default_factory=dict)
    done: bool = False

    def __post_init__(self):
        self.instruction = str(self.instruction).strip() or "stand"
        skill = str(self.skill).strip().lower() or "hold"
        if skill not in SKILL_TO_I:
            skill = "hold"
        self.skill = skill
        raw = self.params if isinstance(self.params, dict) else {}
        clean = {}
        for key, val in raw.items():
            k = str(key).strip().lower()
            if k in {"cube", "reach_cube", "xyz", "object_xyz"}:
                continue
            clean[k] = val
        self.params = clean
        self.done = bool(self.done)
        self._teacher = decode_teacher(self.skill, self.params)

    def requested_yaw(self) -> float | None:
        return parse_requested_yaw(self.skill, self.params)

    def teacher(self) -> TeacherIntent:
        return self._teacher

    def z(self) -> np.ndarray:
        v = np.zeros(Z_DIM, dtype=np.float32)
        v[int(SKILL_TO_I.get(self.skill, 0))] = 1.0
        return v

    def param_text(self) -> str:
        return json.dumps({"skill": self.skill, "params": self.params}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def stand(instruction: str = "stand") -> Plan:
        return Plan(instruction=instruction, skill="stand", params={})


def plan_to_params(plan: Plan) -> dict[str, float]:
    t = plan.teacher()
    req = plan.requested_yaw()
    return {
        "h": float(t.height),
        "vx": float(t.vx),
        "yaw": float(req) if req is not None else 0.0,
        "steps": float(t.steps),
        "wave": float(t.wave),
        "kick": float(t.kick),
        "r_arm": float(t.r_arm),
        "l_arm": float(t.l_arm),
    }


def skill_from_params(params: dict) -> str:
    if isinstance(params, dict) and "skill" in params:
        s = str(params.get("skill", "hold")).lower()
        if s in SKILL_TO_I:
            return s
    if abs(float(params.get("kick", 0.0))) > 0.2:
        return "kick"
    if abs(float(params.get("vx", 0.0))) > 0.08 or float(params.get("steps", 0.0)) >= 1:
        return "locomote"
    if abs(float(params.get("yaw", 0.0))) > 0.08:
        return "turn"
    if float(params.get("wave", 0.0)) > 0.15:
        return "wave"
    if float(params.get("h", 1.0)) < 0.72:
        return "squat"
    if float(params.get("r_arm", 0.0)) > 0.2 or float(params.get("l_arm", 0.0)) > 0.2:
        return "reach"
    return "hold"


def relevant_keys(params: dict) -> tuple[str, ...]:
    keys = tuple(
        k
        for k in PARAM_KEYS
        if abs(float(params.get(k, DEFAULT_PARAMS[k])) - DEFAULT_PARAMS[k]) > 0.08
    )
    return keys or ("h",)


def success_keys(params: dict) -> tuple[str, ...]:
    keys = relevant_keys(params)
    if "steps" in keys and "vx" in keys:
        keys = tuple(k for k in keys if k != "vx")
    return keys or ("h",)


def evaluate_trial(skill: str, params: dict, state: dict) -> tuple[bool, dict]:
    """(success, error_vector) on L1 motor fields. Does not know about the scene.

    Manipulation extras (object pose, etc.) may live on `state` for a skill-specific
    branch below; they never enter `error_vector` or the trial token.
    """
    _ = skill
    achieved = {k: 0.0 for k in PARAM_KEYS}
    for k in PARAM_KEYS:
        if k in state:
            achieved[k] = float(state[k])
    error_vector = {k: 0.0 for k in PARAM_KEYS}
    active = relevant_keys(params)
    for k in active:
        error_vector[k] = (achieved[k] - float(params.get(k, DEFAULT_PARAMS[k]))) / PARAM_SCALE[k]
    if skill in {"reach", "hold"} and "object" in state:
        _ = state["object"]
    fell = bool(state.get("fell", False))
    keys = success_keys(params)
    success = (not fell) and all(abs(error_vector[k]) <= PARAM_OK[k] for k in keys)
    return success, error_vector
