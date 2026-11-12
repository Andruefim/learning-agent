"""Continuum Memory System (CMS) blocks as sparse MoEs with masked low-rank expert slots.
Paper refs: arXiv:2606.03979 §2.2 (CMS, Eq. 1-2), §3.2 (parameter expansion), §3.3 (masking).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    """One CMS level: residual MoE MLP with `n_base` dense experts + `n_slots` low-rank experts."""

    def __init__(self, d: int, n_base: int = 2, n_slots: int = 8, rank: int = 16, hidden: int = 4, top_k: int = 2):
        super().__init__()
        self.d, self.n_base, self.n_slots, self.top_k = d, n_base, n_slots, top_k
        self.norm = nn.LayerNorm(d)
        self.base = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, hidden * d), nn.GELU(), nn.Linear(hidden * d, d)) for _ in range(n_base)]
        )
        self.A = nn.Parameter(torch.randn(n_slots, d, rank) * (d ** -0.5))
        self.B = nn.Parameter(torch.zeros(n_slots, rank, d))
        self.router = nn.Linear(d, n_base + n_slots)
        self.register_buffer("slot_active", torch.zeros(n_slots, dtype=torch.bool))
        self.random_expert = False
        self.focus_slot: int | None = None

    @torch.no_grad()
    def activate_slot(self, init_ab=None) -> int:
        free = (~self.slot_active).nonzero()
        if len(free) == 0:
            norms = self.A.detach().norm(dim=(1, 2)) * self.B.detach().norm(dim=(1, 2))
            i = int(norms.argmin())
        else:
            i = int(free[0])
        if init_ab is not None:
            self.A[i].copy_(init_ab[0])
            self.B[i].copy_(init_ab[1])
        else:
            self.A[i].normal_(0, self.d ** -0.5)
            self.B[i].normal_(0, 1e-3)
        self.router.weight[self.n_base + i].normal_(0, 0.02)
        self.router.bias[self.n_base + i].fill_(0.25)
        self.slot_active[i] = True
        return i

    @torch.no_grad()
    def reset_slots(self) -> None:
        """Synaptic pruning (§3.3): reset transient low-rank experts of this block."""
        self.A.normal_(0, self.d ** -0.5)
        self.B.zero_()
        self.slot_active.zero_()
        self.router.weight[self.n_base :].normal_(0, 0.02)
        self.router.bias[self.n_base :].zero_()
        self.focus_slot = None

    def expert_mask(self) -> torch.Tensor:
        return torch.cat(
            [torch.ones(self.n_base, dtype=torch.bool, device=self.slot_active.device), self.slot_active]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        mask = self.expert_mask()
        logits = self.router(h).masked_fill(~mask, float("-inf"))
        n_act = int(mask.sum())
        k = min(self.top_k + int(self.slot_active.sum()), n_act)
        topv, topi = logits.topk(k, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, topi, F.softmax(topv, dim=-1))

        if self.focus_slot is not None and self.slot_active[self.focus_slot]:
            fidx = self.n_base + self.focus_slot
            f_gate = torch.sigmoid(logits[:, fidx : fidx + 1]) * 0.25
            gates = gates * (1.0 - f_gate)
            gates = gates.scatter_add(
                -1, torch.full((x.shape[0], 1), fidx, dtype=torch.long, device=x.device), f_gate
            )

        if self.random_expert:
            avail = mask.nonzero().squeeze(-1)
            ridx = avail[torch.randint(len(avail), (x.shape[0],), device=x.device)]
            w = 1.0 / (k + 1.0)
            gates = gates * (1.0 - w)
            gates = gates.scatter_add(
                -1, ridx[:, None], torch.full((x.shape[0], 1), w, device=x.device)
            )

        outs = [e(h) for e in self.base]
        slot_out = torch.einsum("bd,sdr->bsr", h, self.A)
        slot_out = torch.einsum("bsr,srd->bsd", F.gelu(slot_out), self.B)
        allout = torch.cat([torch.stack(outs, dim=1), slot_out], dim=1)
        return x + (gates.unsqueeze(-1) * allout).sum(dim=1)

    def base_params(self):
        ps = [p for e in self.base for p in e.parameters()] + list(self.norm.parameters())
        return ps + [self.router.weight, self.router.bias]


class CMSChain(nn.Module):
    """Multi-frequency Continuum Memory Stream chain."""

    def __init__(self, d: int, chunks=(8, 24, 72), n_slots=(8, 8, 24), **kw):
        super().__init__()
        assert all(chunks[i + 1] % chunks[i] == 0 for i in range(len(chunks) - 1))
        self.chunks = list(chunks)
        self.blocks = nn.ModuleList([MoEBlock(d, n_slots=n, **kw) for n in n_slots])

    def set_random_expert(self, on: bool) -> None:
        for b in self.blocks:
            b.random_expert = on

    def used_slots(self) -> int:
        return sum(int(b.slot_active.sum().item()) for b in self.blocks)

    def total_slots(self) -> int:
        return sum(int(b.n_slots) for b in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x
