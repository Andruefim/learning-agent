"""Foundation Vision-Language-Action (VLA) Policy with Continual Memory Stream (CMS)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from agent.config import SLEEP_ACTION_DIM, SLEEP_HORIZON, SLEEP_PROPRIO_DIM
from .cms import CMSChain

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _hash_tok(piece: str, vocab: int) -> int:
    return int(hashlib.md5(piece.encode("utf-8")).hexdigest(), 16) % (vocab - 1) + 1


def tokenize(text: str, vocab: int = 2048, maxlen: int = 24) -> list[int]:
    """Subword-aware instruction tokenizer (handles both RU & EN morphology)."""
    words = _TOKEN_RE.findall(text.lower())
    ids: list[int] = []
    for w in words:
        if len(w) >= 4:
            ids.append(_hash_tok("p:" + w[:4], vocab))
        ids.append(_hash_tok("w:" + w, vocab))
        if len(ids) >= maxlen:
            break
    ids = ids[:maxlen]
    return ids + [0] * (maxlen - len(ids))


class FoundationVLAWithCMS(nn.Module):
    """System 1 fast VLA policy with Gaussian action uncertainty sigma & CMS slots."""

    def __init__(
        self,
        act_dim: int = SLEEP_ACTION_DIM,
        state_dim: int = SLEEP_PROPRIO_DIM,
        horizon: int = SLEEP_HORIZON,
        d: int = 256,
        chunks: tuple[int, ...] = (8, 24, 72),
        n_slots: tuple[int, ...] = (8, 8, 24),
        rank: int = 16,
        vocab: int = 2048,
    ) -> None:
        super().__init__()
        self.A = act_dim
        self.H = horizon
        self.d = d
        self.vocab = vocab

        try:
            base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            base_resnet = models.resnet18(weights=None)
        self.visual_backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        # Freeze early conv layers to keep universal visual features while fine-tuning high-level semantics
        for param in list(self.visual_backbone.parameters())[:25]:
            param.requires_grad = False

        self.visual_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.lang_embed = nn.Embedding(vocab, d, padding_idx=0)
        lang_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=512,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.lang_transformer = nn.TransformerEncoder(lang_layer, num_layers=2)
        self.lang_proj = nn.Linear(d, d)

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.fusion_norm = nn.LayerNorm(d)
        self.cms = CMSChain(d, chunks=chunks, n_slots=n_slots, rank=rank)

        self.head = nn.Sequential(
            nn.Linear(d + horizon * act_dim + horizon, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 2 * act_dim),
        )

    @property
    def vision(self) -> nn.ModuleList:
        return nn.ModuleList([self.visual_backbone, self.visual_proj])

    @property
    def lang(self) -> nn.ModuleList:
        return nn.ModuleList([self.lang_embed, self.lang_transformer, self.lang_proj])

    @property
    def state(self) -> nn.Module:
        return self.state_proj

    def encode(self, img: torch.Tensor, state: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        v_emb = self.visual_proj(self.visual_backbone(img))
        l_tokens = self.lang_embed(tok)
        l_mask = tok == 0
        l_trans = self.lang_transformer(l_tokens, src_key_padding_mask=l_mask)
        lens = (~l_mask).sum(dim=-1, keepdim=True).clamp_min(1)
        l_emb = self.lang_proj(l_trans.sum(dim=1) / lens)
        s_emb = self.state_proj(state)
        fused = self.fusion_norm(v_emb + l_emb + s_emb)
        return self.cms(fused)

    def step_dist(self, feat: torch.Tensor, prefix: torch.Tensor, h: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feat.shape[0]
        oh = F.one_hot(torch.full((B,), h, device=feat.device, dtype=torch.long), self.H).float()
        head_in = torch.cat([feat, prefix.clamp(-1.5, 1.5), oh], dim=-1)
        mu, ls = self.head(head_in).chunk(2, dim=-1)
        return mu, ls.clamp(-2.8, 0.6)

    def dists(self, feat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mus, lss = [], []
        B = feat.shape[0]
        for h in range(self.H):
            prefix = torch.zeros(B, self.H * self.A, device=feat.device)
            if h > 0:
                prefix[:, : h * self.A] = y[:, :h].reshape(B, -1)
            mu, ls = self.step_dist(feat, prefix, h)
            mus.append(mu)
            lss.append(ls)
        return torch.stack(mus, dim=1), torch.stack(lss, dim=1)

    def nll(self, feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, ls = self.dists(feat, y)
        err2 = (y - mu) ** 2
        var = ls.exp() ** 2 + 1e-4
        nll = 0.5 * (err2 / var) + ls + 0.5 * math.log(2.0 * math.pi)
        return nll.sum(dim=(-1, -2)).mean() + 12.0 * err2[:, 0].sum(dim=-1).mean()

    def sample(
        self,
        feat: torch.Tensor,
        prefix_actions: Optional[torch.Tensor] = None,
        greedy: bool = False,
        temp: float = 1.0,
        return_sigma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, float]:
        B = feat.shape[0]
        y = torch.zeros(B, self.H, self.A, device=feat.device)
        logp = torch.zeros(B, device=feat.device)
        sigmas: list[float] = []

        p = 0 if prefix_actions is None else prefix_actions.shape[1]
        if p > 0:
            y[:, :p] = prefix_actions

        for h in range(p, self.H):
            prefix = torch.zeros(B, self.H * self.A, device=feat.device)
            if h > 0:
                prefix[:, : h * self.A] = y[:, :h].reshape(B, -1).detach()
            mu, ls = self.step_dist(feat, prefix, h)
            std = ls.exp().clamp_min(0.08)
            sigmas.append(float(std.mean().item()))
            if greedy:
                a = mu.clamp(-1.2, 1.2)
            else:
                a = (mu + std * (torch.randn_like(mu) * temp)).clamp(-1.2, 1.2)
            logp = logp + (-0.5 * ((a - mu) / std) ** 2 - ls - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
            y[:, h] = a

        mean_sigma = float(sum(sigmas) / max(len(sigmas), 1))
        if return_sigma:
            return y, logp, mean_sigma
        return y, logp

    def predict_action(self, feat: torch.Tensor, greedy: bool = True) -> Tuple[torch.Tensor, float]:
        y, _, sigma = self.sample(feat, greedy=greedy, return_sigma=True)
        return y, sigma


def gaussian_kl(mu_t: torch.Tensor, ls_t: torch.Tensor, mu_s: torch.Tensor, ls_s: torch.Tensor) -> torch.Tensor:
    return (ls_s - ls_t + (torch.exp(2.0 * ls_t) + (mu_t - mu_s) ** 2) / (2.0 * torch.exp(2.0 * ls_s)) - 0.5).sum(
        dim=(-1, -2)
    )
