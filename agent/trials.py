"""In-context trial tokens. Correction is the FlowPolicy forward, not this buffer."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import torch

from agent.config import PARAM_KEYS, PARAM_SCALE, SKILL_IDS, SKILL_TO_I, STAND_Z, TRIAL_MAX
from agent.h2 import ACTION_DIM, N_ACT, TRIAL_FEAT
from agent.plan import success_keys


@dataclass
class Trial:
    skill: str
    params: dict
    achieved: dict
    error_vector: dict
    fell: bool
    success: bool
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)

    def log_line(self, n: int) -> str:
        bits = []
        for k in success_keys(self.params):
            raw = float(self.error_vector.get(k, 0.0)) * PARAM_SCALE[k]
            if k == "h":
                bits.append(f"Δheight={raw * STAND_Z:+.2f}м")
            elif k == "steps":
                bits.append(f"Δsteps={raw:+.0f}")
            else:
                bits.append(f"Δ{k}={raw:+.2f}")
        body = " · ".join(bits) if bits else "ok"
        extra = " · fell" if self.fell else ""
        return f"Попытка {n}: skill={self.skill} · {body}{extra}"

    def as_public(self, n: int) -> dict:
        return {
            "n": n,
            "skill": self.skill,
            "success": self.success,
            "fell": self.fell,
            "line": self.log_line(n),
            "error": {k: round(float(self.error_vector.get(k, 0.0)), 3) for k in PARAM_KEYS},
        }


class MultiTrialBuffer:
    def __init__(self, maxlen: int = TRIAL_MAX):
        self.maxlen = int(maxlen)
        self._items: collections.deque[Trial] = collections.deque(maxlen=self.maxlen)

    def clear(self):
        self._items.clear()

    def append(self, trial: Trial):
        self._items.append(trial)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def items(self) -> list[Trial]:
        return list(self._items)

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.full((1, self.maxlen), len(SKILL_IDS), dtype=torch.long, device=device)
        feat = torch.zeros(1, self.maxlen, TRIAL_FEAT, device=device, dtype=torch.float32)
        for i, trial in enumerate(self._items):
            ids[0, i] = int(SKILL_TO_I.get(trial.skill, 0))
            state = list(trial.state or [])[:N_ACT] + [0.0] * N_ACT
            action = list(trial.action or [])[:ACTION_DIM] + [0.0] * ACTION_DIM
            err = [float(trial.error_vector.get(k, 0.0)) for k in PARAM_KEYS]
            row = state[:N_ACT] + action[:ACTION_DIM] + err
            feat[0, i] = torch.tensor(row[:TRIAL_FEAT], device=device, dtype=torch.float32)
        return ids, feat
