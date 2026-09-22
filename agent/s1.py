"""Causal System 1. Predicts the next 18-D command from the command history.

Awake it only runs forward. Sleep fits the next command on replay. The walk
policy and the teacher residual stay outside this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agent.config import (
    CHUNK,
    ERROR_LEN,
    LANG_DIM,
    SKILL_IDS,
    TRIAL_EMB,
    TRIAL_MAX,
    Z_DIM,
)
from agent.h2 import ACTION_DIM, TRIAL_FEAT
from agent.policy import LanguageEncoder

N_RAYS = 5
HIST_DIM = ACTION_DIM + 3 + N_RAYS
CTX = 16
D_MODEL = 128
N_LAYERS = 4


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        t = h.shape[1]
        mask = torch.triu(torch.ones(t, t, device=h.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class CommandTransformer(nn.Module):
    """Next command, one frame at a time. Rays start at zero weight until sleep sees them."""

    def __init__(self):
        super().__init__()
        self.lang = LanguageEncoder()
        self.skill_emb = nn.Embedding(len(SKILL_IDS) + 1, 8)
        self.trial_mlp = nn.Sequential(
            nn.Linear(8 + TRIAL_FEAT, TRIAL_EMB),
            nn.SiLU(),
            nn.Linear(TRIAL_EMB, TRIAL_EMB),
        )
        prefix_in = LANG_DIM + Z_DIM + ERROR_LEN * 3 + TRIAL_EMB
        self.prefix = nn.Sequential(nn.Linear(prefix_in, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, D_MODEL))
        self.cmd_proj = nn.Linear(HIST_DIM, D_MODEL)
        self.pos = nn.Embedding(CTX + CHUNK + 4, D_MODEL)
        self.blocks = nn.ModuleList(_Block(D_MODEL) for _ in range(N_LAYERS))
        self.head = nn.Linear(D_MODEL, ACTION_DIM)
        nn.init.zeros_(self.cmd_proj.weight[:, ACTION_DIM + 3 :])
        nn.init.zeros_(self.cmd_proj.bias)

    def encode_trials(self, skill_ids, trial_feat):
        sk = self.skill_emb(skill_ids.clamp(0, len(SKILL_IDS)))
        return self.trial_mlp(torch.cat([sk, trial_feat], dim=-1))

    def _trial_pad(self, image):
        b = image.shape[0]
        ids = torch.full((b, TRIAL_MAX), len(SKILL_IDS), device=image.device, dtype=torch.long)
        feat = torch.zeros(b, TRIAL_MAX, TRIAL_FEAT, device=image.device, dtype=image.dtype)
        return ids, feat

    def _prefix(self, image, language, z, errors, skill_ids, trial_feat) -> torch.Tensor:
        if skill_ids is None or trial_feat is None:
            skill_ids, trial_feat = self._trial_pad(image)
        trials = self.encode_trials(skill_ids, trial_feat).mean(dim=1)
        flat = torch.cat([self.lang(language), z, errors.flatten(1), trials], dim=-1)
        return self.prefix(flat)

    def _run(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        pos = self.pos(torch.arange(t, device=tokens.device)).unsqueeze(0)
        h = tokens + pos
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def sample(
        self,
        image,
        proprio,
        language,
        z,
        errors,
        skill_ids=None,
        trial_feat=None,
        history: torch.Tensor | None = None,
        steps: int = CHUNK,
    ):
        del proprio
        prefix = self._prefix(image, language, z, errors, skill_ids, trial_feat)
        b = prefix.shape[0]
        if history is None:
            err = errors[:, :, :3]
            raw = prefix.new_zeros(b, err.shape[1], HIST_DIM)
            raw[:, :, ACTION_DIM : ACTION_DIM + 3] = err
        else:
            raw = history
            if raw.shape[0] == 1 and b > 1:
                raw = raw.expand(b, -1, -1)
        seq = torch.cat([prefix.unsqueeze(1), self.cmd_proj(raw)], dim=1)
        outs = []
        for _ in range(int(steps)):
            nxt = self._run(seq)[:, -1]
            outs.append(nxt)
            tok = prefix.new_zeros(b, 1, HIST_DIM)
            tok[:, 0, :ACTION_DIM] = nxt
            seq = torch.cat([seq, self.cmd_proj(tok)], dim=1)
        return torch.stack(outs, dim=1)

    def cfm_loss(
        self,
        image,
        proprio,
        language,
        z,
        errors,
        chunk,
        skill_ids=None,
        trial_feat=None,
        rays: torch.Tensor | None = None,
    ):
        """Teacher-forced next command. `chunk` is (B, T, 18)."""
        del proprio
        prefix = self._prefix(image, language, z, errors, skill_ids, trial_feat)
        prev = chunk[:, :-1]
        err = errors[:, -1, :3].unsqueeze(1).expand(-1, prev.shape[1], -1)
        raw = prefix.new_zeros(prev.shape[0], prev.shape[1], HIST_DIM)
        raw[:, :, :ACTION_DIM] = prev
        raw[:, :, ACTION_DIM : ACTION_DIM + 3] = err
        if rays is not None:
            raw[:, :, ACTION_DIM + 3 :] = rays[:, : prev.shape[1]]
        seq = torch.cat([prefix.unsqueeze(1), self.cmd_proj(raw)], dim=1)
        return nn.functional.mse_loss(self._run(seq), chunk)
