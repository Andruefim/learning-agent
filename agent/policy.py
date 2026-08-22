"""L2 FlowPolicy. Runtime is eval(); weights update only on Save/Consolidate."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent.config import (
    CHUNK,
    ERROR_LEN,
    FLOW_STEPS,
    INSTR_BYTES,
    LANG_DIM,
    SKILL_IDS,
    TRIAL_EMB,
    TRIAL_MAX,
    VISION_DIM,
    Z_DIM,
)
from agent.h2 import ACTION_DIM, N_ACT, TRIAL_FEAT


def resolve_device(spec: str | None = None) -> torch.device:
    spec = (spec or os.getenv("L2_DEVICE", "cpu")).lower().strip()
    if spec in {"cuda", "gpu", "rocm", "hip"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_instr(text: str) -> np.ndarray:
    raw = (text or "").encode("utf-8")[:INSTR_BYTES]
    out = np.zeros(INSTR_BYTES, dtype=np.int64)
    if raw:
        out[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def load_state(module: nn.Module, path: Path, device: torch.device) -> bool:
    if not path.exists():
        return False
    try:
        blob = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location=device)
    try:
        module.load_state_dict(blob)
    except (RuntimeError, ValueError):
        return False
    return True


class VisionEncoder(nn.Module):
    def __init__(self, out: int = VISION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageEncoder(nn.Module):
    def __init__(self, dim: int = LANG_DIM):
        super().__init__()
        self.emb = nn.Embedding(256, dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(tokens.clamp(0, 255).long()).mean(dim=1))


class FlowPolicy(nn.Module):
    """Maps skill + params + trial tokens → action chunk. Correction lives in this graph."""

    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.lang = LanguageEncoder()
        self.skill_emb = nn.Embedding(len(SKILL_IDS) + 1, 8)
        self.trial_mlp = nn.Sequential(
            nn.Linear(8 + TRIAL_FEAT, TRIAL_EMB),
            nn.SiLU(),
            nn.Linear(TRIAL_EMB, TRIAL_EMB),
        )
        base = VISION_DIM + LANG_DIM + N_ACT + Z_DIM + ERROR_LEN * 3
        self.q_proj = nn.Linear(base, TRIAL_EMB)
        self.attn = nn.MultiheadAttention(TRIAL_EMB, num_heads=4, batch_first=True)
        cond = base + TRIAL_EMB
        self.v = nn.Sequential(
            nn.Linear(CHUNK * ACTION_DIM + 1 + cond, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, CHUNK * ACTION_DIM),
        )

    def encode_trials(self, skill_ids, trial_feat):
        sk = self.skill_emb(skill_ids.clamp(0, len(SKILL_IDS)))
        return self.trial_mlp(torch.cat([sk, trial_feat], dim=-1))

    def _trial_pad(self, image):
        b = image.shape[0]
        ids = torch.full((b, TRIAL_MAX), len(SKILL_IDS), device=image.device, dtype=torch.long)
        feat = torch.zeros(b, TRIAL_MAX, TRIAL_FEAT, device=image.device, dtype=image.dtype)
        return ids, feat

    def cond(self, image, proprio, language, z, errors, skill_ids=None, trial_feat=None):
        if skill_ids is None or trial_feat is None:
            skill_ids, trial_feat = self._trial_pad(image)
        base = torch.cat(
            [self.vision(image), self.lang(language), proprio, z, errors.flatten(1)],
            dim=-1,
        )
        tokens = self.encode_trials(skill_ids, trial_feat)
        query = self.q_proj(base).unsqueeze(1)
        attended, _ = self.attn(query, tokens, tokens, need_weights=False)
        return torch.cat([base, attended.squeeze(1)], dim=-1)

    def velocity(self, x_t, t, cond):
        return self.v(torch.cat([x_t.flatten(1), t, cond], dim=-1)).view(-1, CHUNK, ACTION_DIM)

    def sample(self, image, proprio, language, z, errors, skill_ids=None, trial_feat=None, steps: int = FLOW_STEPS):
        cond = self.cond(image, proprio, language, z, errors, skill_ids, trial_feat)
        x = torch.randn(image.shape[0], CHUNK, ACTION_DIM, device=image.device, dtype=image.dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((image.shape[0], 1), i * dt, device=image.device, dtype=image.dtype)
            x = x + dt * self.velocity(x, t, cond)
        return x

    def cfm_loss(self, image, proprio, language, z, errors, chunk, skill_ids=None, trial_feat=None):
        cond = self.cond(image, proprio, language, z, errors, skill_ids, trial_feat)
        eps = torch.randn_like(chunk)
        t = torch.rand(chunk.shape[0], 1, 1, device=chunk.device, dtype=chunk.dtype)
        x_t = (1.0 - t) * eps + t * chunk
        t_in = t.squeeze(-1)
        return nn.functional.mse_loss(self.velocity(x_t, t_in, cond), chunk - eps)
