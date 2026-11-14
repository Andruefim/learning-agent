"""SLEEP continual learner (arXiv:2606.03979): Wake Eq.2, NREM Consolidation (GKD+LTI), REM Dreaming (ReST-EM)."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import gaussian_kl


@dataclass
class SleepConfig:
    lr: float = 8e-4
    # Knowledge Seeding
    ks_steps: int = 20
    ks_batch: int = 16
    lam: float = 0.5
    alpha: float = 0.5
    gamma: float = 0.5
    z0: float = 0.5
    ks_lr: float = 2e-3
    # Dreaming
    dream_m: int = 8
    dream_topk: int = 2
    dream_b: int = 1
    dream_inner_steps: int = 3
    dream_rest_steps: int = 8
    dream_every_sleep: bool = True


class SemanticRewardModel(nn.Module):
    """Frozen skill classifier trained in Wake and used in Sleep LTI reward."""

    def __init__(self, H: int, A: int, n_tasks: int, d: int = 64):
        super().__init__()
        self.n_tasks = n_tasks
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(H * A, d), nn.GELU(), nn.Linear(d, n_tasks))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def _stable_kl(mu_t: torch.Tensor, ls_t: torch.Tensor, mu_s: torch.Tensor, ls_s: torch.Tensor) -> torch.Tensor:
    return gaussian_kl(mu_t, ls_t.clamp_min(-2.0), mu_s, ls_s.clamp_min(-2.0))


class Learner:
    def __init__(self, model, n_tasks: int = 32, cfg: SleepConfig = SleepConfig(), use_sleep: bool = True, log=print):
        self.m, self.cfg, self.use_sleep, self.log = model, cfg, use_sleep, log
        self.n_tasks = n_tasks
        self.blocks = model.cms.blocks
        self.chunks = model.cms.chunks
        self.k = len(self.blocks)
        if self.use_sleep and not self.blocks[0].slot_active.any():
            self.blocks[0].activate_slot()

        self.groups = [list(b.base_params()) for b in self.blocks]
        enc = [p for p in (list(model.vision.parameters()) + list(model.lang.parameters()) + list(model.state.parameters())) if p.requires_grad]
        if self.use_sleep:
            self.groups[0] += enc + [self.blocks[0].A, self.blocks[0].B] + list(model.head.parameters())
            self.opts = [
                torch.optim.Adam(g, lr=cfg.lr if idx == 0 else cfg.lr * min(math.sqrt(c), 2.5))
                for idx, (g, c) in enumerate(zip(self.groups, self.chunks))
            ]
        else:
            self.groups[0] += enc
            self.groups[-1] += list(model.head.parameters())
            self.opts = [torch.optim.Adam(g, lr=cfg.lr) for g in self.groups]

        self.grad_counts = [0] * self.k
        self.step_i = 0
        self.short_term: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.anchors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.prev_teacher = None
        self.task_slots = [set() for _ in range(self.k)]
        dev = next(model.parameters()).device
        self.rm = SemanticRewardModel(model.H, model.A, n_tasks).to(dev)
        self.rm_opt = torch.optim.Adam(self.rm.parameters(), lr=1e-3)
        self.consolidated_skills: set[str] = set()
        self.skill_lessons: dict[str, list[str]] = {}
        self.stats = {"sleeps": 0, "dreams_kept": 0, "dreams_rewarded": 0, "failure_sleeps": 0}

    def add_lesson(self, skill_key: str, lesson: str, max_lessons: int = 4) -> None:
        text = lesson.strip()
        if not skill_key or not text:
            return
        cur = self.skill_lessons.setdefault(skill_key, [])
        if text not in cur:
            cur.append(text)
            if len(cur) > max_lessons:
                del cur[:-max_lessons]

    def get_lessons(self, skill_key: str) -> list[str]:
        return list(self.skill_lessons.get(skill_key, ()))

    def _prepare_group_step(self, l: int) -> None:
        if not self.use_sleep:
            return
        cnt = max(1, self.grad_counts[l])
        with torch.no_grad():
            for p in self.groups[l]:
                if p.grad is not None:
                    p.grad.div_(cnt)
        self.grad_counts[l] = 0

        if self.prev_teacher is not None and len(self.anchors) > 0:
            img_a, st_a, tok_a = self._sample_from_buf(self.anchors, self.cfg.ks_batch)
            with torch.no_grad():
                tf_a = self.prev_teacher.encode(img_a, st_a, tok_a)
                y_a, _ = self.prev_teacher.sample(tf_a, greedy=True)
                mt_a, lt_a = self.prev_teacher.dists(tf_a, y_a)
            sf_a = self.m.encode(img_a, st_a, tok_a)
            ms_a, ls_a = self.m.dists(sf_a, y_a)
            anc_loss = _stable_kl(mt_a, lt_a, ms_a, ls_a).mean() * 0.25 + 4.0 * ((mt_a - ms_a) ** 2).sum((-1, -2)).mean()
            req_params = [p for p in self.groups[l] if p.requires_grad]
            if req_params and anc_loss.requires_grad:
                grads = torch.autograd.grad(anc_loss, req_params, retain_graph=False, allow_unused=True)
                with torch.no_grad():
                    for p, g in zip(req_params, grads):
                        if g is not None:
                            if p.grad is None:
                                p.grad = g.clone()
                            else:
                                p.grad.add_(g)

        if l == 0:
            b0 = self.blocks[0]
            with torch.no_grad():
                if b0.A.grad is not None:
                    b0.A.grad[~b0.slot_active] = 0
                if b0.B.grad is not None:
                    b0.B.grad[~b0.slot_active] = 0
        else:
            b = self.blocks[l]
            if b.slot_active.any():
                with torch.no_grad():
                    if b.router.weight.grad is not None:
                        b.router.weight.grad[b.n_base :][b.slot_active] = 0
                    if b.router.bias.grad is not None:
                        b.router.bias.grad[b.n_base :][b.slot_active] = 0

    def wake_step(self, batch, *, allow_consolidate: bool = True) -> float:
        img, st, tok, y, tid = batch
        tid = tid.clamp(0, self.n_tasks - 1)
        self.m.train()
        feat = self.m.encode(img, st, tok)
        loss = self.m.nll(feat, y)

        if self.use_sleep:
            ood_n = min(4, tok.shape[0])
            tok_ood = torch.randint(100, max(200, self.m.vocab // 2), tok[:ood_n].shape, device=tok.device)
            feat_ood = self.m.encode(img[:ood_n].detach(), st[:ood_n].detach(), tok_ood)
            prefix_z = torch.zeros(feat_ood.shape[0], self.m.H * self.m.A, device=tok.device)
            _, ls_ood = self.m.step_dist(feat_ood, prefix_z, 0)
            loss = loss + 0.15 * ((ls_ood - 0.10) ** 2).mean()

        loss.backward()
        for l in range(self.k):
            self.grad_counts[l] += 1

        self.short_term.append((img.detach(), st.detach(), tok.detach()))
        self.short_term = self.short_term[-64:]

        rl = F.cross_entropy(self.rm(y), tid)
        self.rm_opt.zero_grad()
        rl.backward()
        self.rm_opt.step()

        self.step_i += 1
        for l in range(self.k):
            if self.step_i % self.chunks[l] == 0:
                if self.use_sleep and allow_consolidate and l < self.k - 1:
                    self.consolidate(sender=l, receiver=l + 1)
                else:
                    self._prepare_group_step(l)
                    self.opts[l].step()
                self.opts[l].zero_grad(set_to_none=True)
            elif l == 0 and self.use_sleep:
                self._prepare_group_step(0)
                self.opts[0].step()
                self.opts[0].zero_grad(set_to_none=True)
        return float(loss.detach())

    def _sample_from_buf(self, buf, n: int):
        sel = random.sample(buf, min(len(buf), 8))
        img, st, tok = (torch.cat(z, dim=0) for z in zip(*sel))
        if img.shape[0] > n:
            idx = torch.randperm(img.shape[0], device=img.device)[:n]
            return img[idx], st[idx], tok[idx]
        return img, st, tok

    def consolidate(self, sender: int, receiver: int) -> None:
        c, m = self.cfg, self.m
        if not self.short_term:
            return
        self._prepare_group_step(sender)
        self.opts[sender].step()

        sb = m.cms.blocks[sender]
        rb = m.cms.blocks[receiver]
        init_ab = None
        if sb.slot_active.any() and sb.A.shape[1:] == rb.A.shape[1:]:
            with torch.no_grad():
                act_idx = sb.slot_active.nonzero().squeeze(-1)
                init_ab = (sb.A[act_idx].mean(0).clone(), sb.B[act_idx].mean(0).clone() * 0.25)

        teacher = copy.deepcopy(m).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        sb.reset_slots()
        if sender == 0:
            sb.activate_slot()
        slot = rb.activate_slot(init_ab=init_ab)
        self.task_slots[receiver].add(slot)
        rb.focus_slot = slot

        frozen_A, frozen_B = rb.A.detach().clone(), rb.B.detach().clone()
        rw, rbias = rb.router.weight.detach().clone(), rb.router.bias.detach().clone()
        keep = torch.ones(rb.n_slots, dtype=torch.bool, device=rb.A.device)
        keep[slot] = False
        rmask = torch.ones(rb.n_base + rb.n_slots, dtype=torch.bool, device=rb.A.device)
        rmask[rb.n_base + slot] = False

        train_ps = [rb.A, rb.B, rb.router.weight, rb.router.bias]
        saved_rg = {id(p): p.requires_grad for p in m.parameters()}
        for p in m.parameters():
            p.requires_grad_(False)
        for p in train_ps:
            p.requires_grad_(True)
        opt = torch.optim.Adam(train_ps, lr=c.ks_lr)
        has_anchors = len(self.anchors) > 0 and self.prev_teacher is not None
        m.eval()

        gkd_val, r_val = 0.0, 0.0
        for it in range(c.ks_steps):
            n_cur = c.ks_batch // 2 if has_anchors else c.ks_batch
            img_c, st_c, tok_c = self._sample_from_buf(self.short_term, n_cur)
            with torch.no_grad():
                tf_c = teacher.encode(img_c, st_c, tok_c)
                y_tc, _ = teacher.sample(tf_c, greedy=(it % 2 == 0), temp=0.5)
                mt_c, lt_c = teacher.dists(tf_c, y_tc)

            if has_anchors:
                n_anc = max(4, c.ks_batch - img_c.shape[0])
                img_a, st_a, tok_a = self._sample_from_buf(self.anchors, n_anc)
                with torch.no_grad():
                    tf_a = self.prev_teacher.encode(img_a, st_a, tok_a)
                    y_ta, _ = self.prev_teacher.sample(tf_a, greedy=(it % 2 == 0), temp=0.5)
                    mt_a, lt_a = self.prev_teacher.dists(tf_a, y_ta)
                img = torch.cat([img_c, img_a], 0)
                st = torch.cat([st_c, st_a], 0)
                tok = torch.cat([tok_c, tok_a], 0)
                y_t = torch.cat([y_tc, y_ta], 0)
                mt_t = torch.cat([mt_c, mt_a], 0)
                lt_t = torch.cat([lt_c, lt_a], 0)
            else:
                img, st, tok, y_t, mt_t, lt_t = img_c, st_c, tok_c, y_tc, mt_c, lt_c

            sf = m.encode(img, st, tok)
            with torch.no_grad():
                y_s, _ = m.sample(sf, temp=0.5)

            ms_t, ls_t = m.dists(sf, y_t)
            gkd_off = _stable_kl(mt_t, lt_t, ms_t, ls_t).mean() * 0.25 + 4.0 * ((mt_t - ms_t) ** 2).sum((-1, -2)).mean()

            with torch.no_grad():
                if has_anchors:
                    nt = img_c.shape[0]
                    mt_sc, lt_sc = teacher.dists(tf_c, y_s[:nt])
                    mt_sa, lt_sa = self.prev_teacher.dists(tf_a, y_s[nt:])
                    mt_s = torch.cat([mt_sc, mt_sa], 0)
                    lt_s = torch.cat([lt_sc, lt_sa], 0)
                else:
                    mt_s, lt_s = teacher.dists(tf_c, y_s)
            ms_s, ls_s = m.dists(sf, y_s)
            gkd_on = _stable_kl(mt_s, lt_s, ms_s, ls_s).mean() * 0.25 + 4.0 * ((mt_s - ms_s) ** 2).sum((-1, -2)).mean()
            gkd = (1 - c.lam) * gkd_off + c.lam * gkd_on

            p = random.randint(1, m.H - 1)
            y_hat, logp = m.sample(sf, prefix_actions=y_t[:, :p], temp=0.5)
            with torch.no_grad():
                dist = (y_hat[:, p:] - y_t[:, p:]).abs().sum((-1, -2))
                norm = torch.maximum(y_hat[:, p:].abs().sum((-1, -2)), y_t[:, p:].abs().sum((-1, -2))).clamp_min(1e-6)
                z = dist / norm
                r_abs = torch.where(z <= c.z0, 1 - z, torch.zeros_like(z))
                r_sem = (self.rm(y_hat).argmax(-1) == self.rm(y_t).argmax(-1)).float()
                r = c.gamma * r_sem + (1 - c.gamma) * r_abs
                adv = r - r.mean()
            lti = -(adv * logp.clamp(-20.0, 20.0)).mean()
            loss = (1 - c.alpha) * lti + c.alpha * gkd
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                rb.A.grad[keep] = 0
                rb.B.grad[keep] = 0
                rb.router.weight.grad[rmask] = 0
                rb.router.bias.grad[rmask] = 0
            opt.step()
            with torch.no_grad():
                rb.A[keep] = frozen_A[keep]
                rb.B[keep] = frozen_B[keep]
                rb.router.weight[rmask] = rw[rmask]
                rb.router.bias[rmask] = rbias[rmask]
            gkd_val = float(gkd.detach())
            r_val = float(r.mean().detach())

        rb.focus_slot = None
        for p in m.parameters():
            p.requires_grad_(saved_rg[id(p)])
        self.stats["sleeps"] += 1
        self.log(f"[NREM] Консолидация L{sender}->L{receiver} (слот #{slot}) · gkd={gkd_val:.3f} r={r_val:.3f}")

    def _sft(self, model, img, st, tok, y, steps: int, lr: float) -> None:
        ps = [b.A for b in model.cms.blocks] + [b.B for b in model.cms.blocks]
        frozen = [(b.A.detach().clone(), b.B.detach().clone()) for b in model.cms.blocks]
        opt = torch.optim.Adam(ps, lr=lr)
        for _ in range(steps):
            loss = model.nll(model.encode(img, st, tok), y)
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                for l, b in enumerate(model.cms.blocks):
                    allow = torch.zeros(b.n_slots, dtype=torch.bool, device=b.A.device)
                    for s_idx in self.task_slots[l]:
                        if s_idx < b.n_slots and b.slot_active[s_idx]:
                            allow[s_idx] = True
                    if l == 0 and b.slot_active.any():
                        allow |= b.slot_active
                    b.A.grad[~allow] = 0
                    b.B.grad[~allow] = 0
            opt.step()
            with torch.no_grad():
                for l, b in enumerate(model.cms.blocks):
                    allow = torch.zeros(b.n_slots, dtype=torch.bool, device=b.A.device)
                    for s_idx in self.task_slots[l]:
                        if s_idx < b.n_slots and b.slot_active[s_idx]:
                            allow[s_idx] = True
                    if l == 0 and b.slot_active.any():
                        allow |= b.slot_active
                    b.A[~allow] = frozen[l][0][~allow]
                    b.B[~allow] = frozen[l][1][~allow]

    def dream(self, context, tau) -> None:
        c, m = self.cfg, self.m
        if not any(b.slot_active.any() for b in self.blocks):
            return
        img, st, tok = context
        m.eval()
        with torch.no_grad():
            y_ref, _ = m.sample(m.encode(img, st, tok), greedy=True)
        m.zero_grad(set_to_none=True)
        m.nll(m.encode(img, st, tok), y_ref).backward()
        slot_ps = [b.A for b in self.blocks] + [b.B for b in self.blocks]
        g_ref = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in slot_ps])
        g_ref_norm = g_ref.norm().clamp_min(1e-8)
        m.zero_grad(set_to_none=True)

        m.cms.set_random_expert(True)
        with torch.no_grad():
            dreams = [m.sample(m.encode(img, st, tok), temp=0.25)[0] for _ in range(c.dream_m)]
        m.cms.set_random_expert(False)

        scores = []
        for y in dreams:
            m.zero_grad(set_to_none=True)
            m.nll(m.encode(img, st, tok), y).backward()
            g_i = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in slot_ps])
            cos_sim = float(torch.dot(g_i, g_ref) / (g_i.norm().clamp_min(1e-8) * g_ref_norm))
            scores.append(cos_sim)
        m.zero_grad(set_to_none=True)

        order = sorted(range(len(dreams)), key=lambda i: -scores[i])
        keep = order[: c.dream_topk]
        rest = order[c.dream_topk :]
        keep += random.sample(rest, min(c.dream_b, len(rest)))

        base_score = tau(m)
        good = []
        for i in keep:
            mi = copy.deepcopy(m)
            self._sft(mi, img, st, tok, dreams[i], c.dream_inner_steps, c.ks_lr * 0.5)
            if tau(mi) > base_score:
                good.append(i)
        self.stats["dreams_kept"] += len(keep)
        self.stats["dreams_rewarded"] += len(good)
        if good:
            y = torch.cat([dreams[i] for i in good], dim=0)
            n = len(good)
            self._sft(m, img.repeat(n, 1, 1, 1), st.repeat(n, 1), tok.repeat(n, 1), y, c.dream_rest_steps, c.ks_lr * 0.5)
        self.log(f"[REM Сновидения] сгенерировано={len(dreams)}, отобрано={len(keep)}, улучшили={len(good)}")

    def rem_failure_sleep(
        self, fail_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], steps: int = 3
    ) -> dict[str, float]:
        """Post-failure REM exploration: push L0 away from failed actions and keep high epistemic σ."""
        c, m = self.cfg, self.m
        img, st, tok, y_bad = fail_batch
        if img.shape[0] == 0:
            return {"sigma": 0.85, "sep": 0.0}

        if not self.blocks[0].slot_active.any():
            self.blocks[0].activate_slot()

        m.eval()
        m.cms.set_random_expert(True)
        with torch.no_grad():
            f0 = m.encode(img, st, tok)
            cands = [m.sample(f0, temp=0.45)[0] for _ in range(max(4, c.dream_m // 2))]
        m.cms.set_random_expert(False)

        best_cand = None
        best_sep = -1e9
        for y_c in cands:
            sep_arm = float(((y_c[:, :, 2:] - y_bad[:, :, 2:]) ** 2).mean().clamp(max=0.40).item())
            smooth_arm = float(((y_c[:, 1:, 2:] - y_c[:, :-1, 2:]) ** 2).mean().item()) if y_c.shape[1] > 1 else 0.0
            loc_pen = float(((y_c[:, :, :2] - y_bad[:, :, :2]) ** 2).mean().item())
            score = sep_arm - 0.35 * smooth_arm - 0.25 * loc_pen
            if score > best_sep:
                best_sep = score
                best_cand = y_c

        saved_rg = {id(p): p.requires_grad for p in m.parameters()}
        for p in m.parameters():
            p.requires_grad_(False)
        b0 = self.blocks[0]
        train_ps = [b0.A, b0.B] + list(m.head.parameters())
        for p in train_ps:
            p.requires_grad_(True)
        opt = torch.optim.Adam(train_ps, lr=c.ks_lr * 0.25)

        has_anchors = len(self.anchors) > 0 and self.prev_teacher is not None
        target_ls = -0.45  # keep std ~ 0.64 on unresolved failed states
        for _ in range(steps):
            sf = m.encode(img, st, tok)
            mu_bad, ls_bad = m.dists(sf, y_bad)
            # Negative-reward separation + uncertainty calibration
            mse_to_bad = ((mu_bad - y_bad) ** 2).mean()
            repel = -mse_to_bad.clamp(max=0.20)
            sigma_cal = ((ls_bad - target_ls) ** 2).mean()
            loss = repel + 0.5 * sigma_cal

            if best_cand is not None:
                mu_alt, _ = m.dists(sf, best_cand)
                loss = loss + 0.15 * ((mu_alt - best_cand) ** 2).mean()

            if has_anchors:
                img_a, st_a, tok_a = self._sample_from_buf(self.anchors, min(8, len(self.anchors)))
                with torch.no_grad():
                    tf_a = self.prev_teacher.encode(img_a, st_a, tok_a)
                    y_a, _ = self.prev_teacher.sample(tf_a, greedy=True)
                    mt_a, lt_a = self.prev_teacher.dists(tf_a, y_a)
                sf_a = m.encode(img_a, st_a, tok_a)
                ms_a, ls_a = m.dists(sf_a, y_a)
                loss = loss + _stable_kl(mt_a, lt_a, ms_a, ls_a).mean() * 0.25 + 2.0 * ((mt_a - ms_a) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            f_after = m.encode(img[:1], st[:1], tok[:1])
            _, sigma_after = m.predict_action(f_after, greedy=True)

        for p in m.parameters():
            p.requires_grad_(saved_rg[id(p)])

        self.stats["sleeps"] += 1
        self.stats["failure_sleeps"] = self.stats.get("failure_sleeps", 0) + 1
        self.stats["dreams_kept"] += len(cands)
        self.log(f"[REM Анализ неудачи] чанков={img.shape[0]} · кандидатов={len(cands)} · σ_raw={sigma_after:.3f} (L1/L2 не затронуты)")
        return {"sigma": float(sigma_after), "sep": float(max(0.0, best_sep))}

    def end_of_task_sleep(self, context=None, tau=None) -> None:
        if not self.use_sleep:
            for o in self.opts:
                o.step()
                o.zero_grad(set_to_none=True)
            return
        for l in range(self.k - 1):
            self.consolidate(l, l + 1)
            self.opts[l].zero_grad(set_to_none=True)
        self._prepare_group_step(self.k - 1)
        self.opts[-1].step()
        self.opts[-1].zero_grad(set_to_none=True)
        if self.cfg.dream_every_sleep and context is not None and tau is not None:
            self.dream(context, tau)
        if self.short_term:
            sampled = random.sample(self.short_term, min(len(self.short_term), 8))
            self.anchors.extend([(im.cpu(), s.cpu(), t.cpu()) for im, s, t in sampled])
            self.anchors = [(im.to(next(self.m.parameters()).device), s.to(next(self.m.parameters()).device), t.to(next(self.m.parameters()).device)) for im, s, t in self.anchors[-32:]]
        self.prev_teacher = copy.deepcopy(self.m).eval()
        for p in self.prev_teacher.parameters():
            p.requires_grad_(False)
        self.short_term.clear()
        self.task_slots = [set() for _ in range(self.k)]

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.m.state_dict(),
            "rm": self.rm.state_dict(),
            "stats": dict(self.stats),
            "step_i": self.step_i,
            "consolidated_skills": sorted(self.consolidated_skills),
            "skill_lessons": dict(self.skill_lessons),
            "anchors": [(im.cpu(), s.cpu(), t.cpu()) for im, s, t in self.anchors[-24:]],
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            dev = next(self.m.parameters()).device
            payload = torch.load(str(path), map_location=dev, weights_only=False)
            self.m.load_state_dict(payload["model"], strict=True)
            if "rm" in payload:
                self.rm.load_state_dict(payload["rm"], strict=False)
            if "stats" in payload:
                self.stats.update(payload["stats"])
            self.step_i = int(payload.get("step_i", 0))
            self.consolidated_skills = set(str(s) for s in payload.get("consolidated_skills", ()))
            raw_lessons = payload.get("skill_lessons") or {}
            if isinstance(raw_lessons, dict):
                self.skill_lessons = {str(k): [str(x) for x in v][-4:] for k, v in raw_lessons.items() if isinstance(v, list)}
            if "anchors" in payload:
                self.anchors = [(im.to(dev), s.to(dev), t.to(dev)) for im, s, t in payload["anchors"]]
            if self.m.cms.used_slots() > 0:
                self.prev_teacher = copy.deepcopy(self.m).eval()
                for p in self.prev_teacher.parameters():
                    p.requires_grad_(False)
            return True
        except Exception as exc:
            self.log(f"[ckpt] Не удалось загрузить {path.name}: {exc}")
            return False
