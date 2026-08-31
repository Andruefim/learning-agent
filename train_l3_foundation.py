"""Playground-style PPO for HumanoidFoundationPolicy.

Prefers MJX-JAX on ROCm (2048+ envs). Falls back to in-process MuJoCo vec envs.
Writes flywheel_data/l3_foundation.pt only if eval_reach, eval_deep_squat, and
eval_push_recovery all pass (honest 15s reach / deep squat / push recovery).

JAX and Torch PPO both use multi-epoch minibatch updates + entropy bonus.
Stage A (STAND_ONLY): curriculum stays at stand — no vx/yaw until the policy
can hold tilt without rocking. Curriculum stages follow absolute global_iter
when STAND_ONLY is flipped off.

Run:  python train_l3_foundation.py
      python train_l3_foundation.py --iters 200 --envs 2048
      python train_l3_foundation.py --iters 200 --envs 2048 --resume
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform in {"win32", "darwin"} else "egl"

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agent.config import ROOT as AGENT_ROOT
from agent.l3_env import (
    STAGE_FULL,
    STAGE_STAND,
    STAGE_VX,
    VecFoundationEnv,
    eval_inplace,
    jax_device_kind,
    jax_import_error,
)
from agent.l3_foundation import (
    ACT_DIM,
    BALANCE_PRIOR_SCALE,
    OBS_DIM,
    PUSH_FORCE,
    REACH_FRAC,
    SQUAT_FRAC,
    STAND_ONLY,
    TRAIN_TILT,
    HumanoidFoundationPolicy,
    jax_params_to_state_dict,
    state_dict_to_jax,
)
from agent.policy import resolve_device


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = HumanoidFoundationPolicy(zero_out=True)
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().clamp(1e-3, 1.0).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


# Absolute curriculum boundaries (independent of --iters / resume length).
# Matches the old 200-iter schedule: stand 1–39, vx 40–79, full from 80.
CURRICULUM_HORIZON = 200
RESUME_EVERY = 25  # .latest snapshot; not the production gate
EVAL_EVERY = 50
CHECKPOINT_VERSION = 2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
PPO_LR = 3e-4
PPO_CLIP = 0.2
PPO_TARGET_KL = 0.02
PPO_GAMMA = 0.99
PPO_LAM = 0.95
JAX_MINIBATCH = 4096
JAX_DEFAULT_ENVS = 2048


def curriculum_stage(global_it: int, horizon: int = CURRICULUM_HORIZON) -> int:
    if STAND_ONLY:
        return STAGE_STAND
    if global_it < max(1, horizon // 5):
        return STAGE_STAND
    if global_it < max(2, (2 * horizon) // 5):
        return STAGE_VX
    return STAGE_FULL


def latest_path_for(out_path: Path) -> Path:
    """Resume checkpoint next to the gated one. Different name so engine.py
    (which only loads `l3_foundation.pt`) never picks this up by accident."""
    return out_path.with_name(f"{out_path.stem}.latest.pt")


def _torch_load(path: Path, device: torch.device):
    """Load a local resume checkpoint. Prefer safe mode; fall back for numpy blobs."""
    kwargs = {"map_location": device}
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc).lower()
        if "weights only" in msg or "weights_only" in msg or "unpickler" in msg or "safe_globals" in msg:
            return torch.load(path, weights_only=False, **kwargs)
        raise


def _is_actor_state_dict(blob: dict) -> bool:
    return any(str(k).startswith("net.") for k in blob)


def load_training_checkpoint(path: Path, device: torch.device) -> dict | None:
    """Load resume blob. Supports v2/v1 (actor+critic+log_std[+global_iter]) and legacy actor-only."""
    if not path.is_file():
        return None
    blob = _torch_load(path, device)
    if not isinstance(blob, dict):
        return None
    ver = blob.get("version")
    if ver in (1, 2, CHECKPOINT_VERSION):
        out: dict = {"actor": blob["actor"], "backend": blob.get("backend"), "global_iter": int(blob.get("global_iter", 0))}
        if "critic" in blob:
            out["critic"] = blob["critic"]
        if "critic_jax" in blob:
            out["critic_jax"] = blob["critic_jax"]
        if "log_std" in blob:
            ls = blob["log_std"]
            out["log_std"] = ls if torch.is_tensor(ls) else torch.as_tensor(np.asarray(ls), dtype=torch.float32)
        return out
    if _is_actor_state_dict(blob):
        return {"actor": blob, "backend": None, "global_iter": 0}
    return None


def apply_critic_jax_to_torch(critic_jax, critic: nn.Sequential) -> None:
    for layer_idx, (w, b) in zip((0, 2, 4), critic_jax):
        critic[layer_idx].weight.data.copy_(torch.as_tensor(np.asarray(w).T, dtype=torch.float32))
        critic[layer_idx].bias.data.copy_(torch.as_tensor(np.asarray(b), dtype=torch.float32))


def apply_resume_to_net(net: ActorCritic, resume: dict) -> str:
    net.actor.load_state_dict(resume["actor"])
    parts = ["actor"]
    if "critic" in resume:
        net.critic.load_state_dict(resume["critic"])
        parts.append("critic")
    elif "critic_jax" in resume:
        apply_critic_jax_to_torch(resume["critic_jax"], net.critic)
        parts.append("critic")
    if "log_std" in resume:
        net.log_std.data.copy_(resume["log_std"].reshape(ACT_DIM))
        parts.append("log_std")
    return "+".join(parts)


def save_training_checkpoint(
    path: Path,
    *,
    actor_sd: dict,
    backend: str,
    global_iter: int,
    critic_sd: dict | None = None,
    critic_jax=None,
    log_std=None,
) -> None:
    """Unconditional PPO snapshot for resume. Not gated, not loaded by the live app."""
    if not all(torch.isfinite(v).all() for v in actor_sd.values() if torch.is_tensor(v)):
        print(f"resume checkpoint skipped (non-finite actor weights) {path}", flush=True)
        return
    payload: dict = {
        "version": CHECKPOINT_VERSION,
        "actor": actor_sd,
        "backend": backend,
        "global_iter": int(global_iter),
    }
    if critic_sd is not None:
        payload["critic"] = critic_sd
    if critic_jax is not None:
        payload["critic_jax"] = [
            (
                torch.as_tensor(np.array(w, copy=True), dtype=torch.float32),
                torch.as_tensor(np.array(b, copy=True), dtype=torch.float32),
            )
            for w, b in critic_jax
        ]
    if log_std is not None:
        if torch.is_tensor(log_std):
            payload["log_std"] = log_std.detach().cpu()
        else:
            payload["log_std"] = torch.as_tensor(np.array(log_std, copy=True), dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"resume checkpoint saved {path} global_iter={global_iter}", flush=True)


def save_if_stand(policy: HumanoidFoundationPolicy, path: Path, device: torch.device) -> bool:
    report = eval_inplace(policy, seconds=4.0, device=device)
    cases = report.get("cases", {})
    bits = " ".join(
        f"{name}={'ok' if c['ok'] else 'fail'}(z={c['z_min']:.2f}/{c.get('z_last', c['z_min']):.2f} "
        f"tilt={c['tilt_min']:.3f}{' fell' if c.get('fell') else ''})"
        for name, c in cases.items()
    )
    gate = ("reach", "deep_squat", "push_recovery")
    print(f"eval suite: {'PASS' if report['ok'] else 'fail'} {bits}", flush=True)
    for name in ("reach", "deep_squat", "push_recovery", "locomotion", "static_60s"):
        c = cases.get(name)
        if not c:
            continue
        print(
            f"metrics {name} z_min={c['z_min']:.3f} z_last={c.get('z_last', c['z_min']):.3f} "
            f"tilt_min={c['tilt_min']:.3f} fell={int(bool(c.get('fell')))} ok={int(bool(c['ok']))}",
            flush=True,
        )
    if not report["ok"]:
        missing = [n for n in gate if not cases.get(n, {}).get("ok")]
        print(f"production gate blocked save ({', '.join(missing) or 'unknown'})", flush=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)
    print(f"saved {path}", flush=True)
    return True


def ppo_torch(
    *,
    net: ActorCritic,
    envs: VecFoundationEnv,
    iters: int,
    unroll: int,
    device: torch.device,
    lr: float,
    gamma: float,
    lam: float,
    clip: float,
    epochs: int,
    minibatch: int,
    target_kl: float,
    out_path: Path,
    start_iter: int = 0,
) -> HumanoidFoundationPolicy:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.as_tensor(envs.reset(), device=device, dtype=torch.float32)
    t0 = time.time()
    n = envs.n
    for local in range(1, iters + 1):
        global_it = start_iter + local
        envs.stage = curriculum_stage(global_it)
        buf_o, buf_a, buf_lp, buf_v, buf_r, buf_d = [], [], [], [], [], []
        env_steps = 0
        for _ in range(unroll):
            with torch.no_grad():
                dist = net.dist(obs)
                raw = dist.sample()
                logp = dist.log_prob(raw).sum(-1)
                val = net.value(obs)
            act = torch.tanh(raw)
            nxt, rew, done = envs.step(act.detach().cpu().numpy())
            buf_o.append(obs.cpu().numpy())
            buf_a.append(raw.cpu().numpy())
            buf_lp.append(logp.cpu().numpy())
            buf_v.append(val.cpu().numpy())
            buf_r.append(rew)
            buf_d.append(done)
            obs = torch.as_tensor(nxt, device=device, dtype=torch.float32)
            env_steps += n
        o = torch.as_tensor(np.stack(buf_o), device=device, dtype=torch.float32)
        a = torch.as_tensor(np.stack(buf_a), device=device, dtype=torch.float32)
        old_lp = torch.as_tensor(np.stack(buf_lp), device=device, dtype=torch.float32)
        v = torch.as_tensor(np.stack(buf_v), device=device, dtype=torch.float32)
        r = torch.as_tensor(np.stack(buf_r), device=device, dtype=torch.float32)
        d = torch.as_tensor(np.stack(buf_d), device=device, dtype=torch.float32)
        with torch.no_grad():
            last_v = net.value(obs)
        adv = torch.zeros_like(r)
        last_gae = torch.zeros(n, device=device)
        for t in reversed(range(unroll)):
            nxt_v = last_v if t == unroll - 1 else v[t + 1]
            mask = 1.0 - d[t]
            delta = r[t] + gamma * nxt_v * mask - v[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        o_f = o.reshape(-1, OBS_DIM)
        a_f = a.reshape(-1, ACT_DIM)
        lp_f = old_lp.reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)
        idx = np.arange(o_f.shape[0])
        for _ in range(epochs):
            np.random.shuffle(idx)
            stop = False
            for s in range(0, idx.size, minibatch):
                mb = idx[s : s + minibatch]
                dist = net.dist(o_f[mb])
                logp = dist.log_prob(a_f[mb]).sum(-1)
                ratio = (logp - lp_f[mb]).exp()
                surr1 = ratio * adv_f[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_f[mb]
                pol = -torch.min(surr1, surr2).mean()
                vf = 0.5 * (net.value(o_f[mb]) - ret_f[mb]).pow(2).mean()
                ent = -ENTROPY_COEF * dist.entropy().sum(-1).mean()
                loss = pol + vf + ent
                approx_kl = float((lp_f[mb] - logp).mean().item())
                if approx_kl > 1.5 * target_kl:
                    stop = True
                    break
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if stop:
                break
        mean_ret = float(r.sum(0).mean().cpu())
        fall_rate = float(d.mean().cpu())
        sps = env_steps / max(time.time() - t0, 1e-6)
        if local == 1 or local % 5 == 0 or local == iters:
            print(
                f"iter {global_it} (+{local}/{iters}) stage={envs.stage} ret={mean_ret:.2f} "
                f"fall_rate={fall_rate:.3f} env-steps/s={sps:.0f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        t0 = time.time()
        if local % RESUME_EVERY == 0 or local == iters:
            save_training_checkpoint(
                latest_path_for(out_path),
                actor_sd=net.actor.state_dict(),
                critic_sd=net.critic.state_dict(),
                log_std=net.log_std,
                backend="torch",
                global_iter=global_it,
            )
        if local == iters or local % EVAL_EVERY == 0:
            save_if_stand(net.actor, out_path, device)
    return net.actor


def ppo_jax(
    *,
    n_envs: int,
    iters: int,
    unroll: int,
    seed: int,
    out_path: Path,
    torch_device: torch.device,
    resume: dict | None = None,
    start_iter: int = 0,
    epochs: int = PPO_EPOCHS,
    minibatch: int = JAX_MINIBATCH,
) -> HumanoidFoundationPolicy:
    import jax
    import jax.numpy as jp

    from agent.l3_mjx import MjxFoundationEnv, mlp_forward, mlp_init

    env = MjxFoundationEnv(n_envs, stage=STAGE_STAND)
    key = jax.random.PRNGKey(seed)
    key, k_p, k_c = jax.random.split(key, 3)
    if resume is not None:
        actor = state_dict_to_jax(resume["actor"])
    else:
        actor = mlp_init(jax, jp, k_p, zero_out=True)

    def init_critic(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return [
            (jax.random.normal(k1, (OBS_DIM, 256)) * jp.sqrt(2.0 / OBS_DIM), jp.zeros((256,))),
            (jax.random.normal(k2, (256, 256)) * jp.sqrt(2.0 / 256), jp.zeros((256,))),
            (jp.zeros((256, 1)), jp.zeros((1,))),
        ]

    if resume is not None and "critic_jax" in resume:
        critic = [(jp.asarray(np.asarray(w)), jp.asarray(np.asarray(b))) for w, b in resume["critic_jax"]]
    elif resume is not None and "critic" in resume:
        critic = init_critic(k_c)
        net_tmp = ActorCritic().to(torch_device)
        net_tmp.critic.load_state_dict(resume["critic"])
        for layer_idx, jax_i in zip((0, 2, 4), range(3)):
            w = net_tmp.critic[layer_idx].weight.detach().cpu().numpy().T
            b = net_tmp.critic[layer_idx].bias.detach().cpu().numpy()
            critic[jax_i] = (jp.asarray(w), jp.asarray(b))
    else:
        critic = init_critic(k_c)

    if resume is not None and "log_std" in resume:
        log_std = jp.asarray(np.asarray(resume["log_std"].detach().cpu().numpy()), dtype=jp.float32)
    else:
        log_std = jp.full((ACT_DIM,), -1.2)
    policy = HumanoidFoundationPolicy(zero_out=True)
    rng = jax.random.PRNGKey(seed + 1)
    # Align env stage with resumed curriculum before first reset.
    env.stage = curriculum_stage(start_iter + 1)
    env._compile()
    obs = env.reset(rng)
    t0 = time.time()
    mb = max(1, min(int(minibatch), n_envs * unroll))
    n_samples = int(n_envs * unroll)
    n_mb = max(1, n_samples // mb)
    step_v = env._step_v

    def critic_apply(params, x):
        h = x
        for w, b in params[:-1]:
            z = h @ w + b
            h = z * jax.nn.sigmoid(z)
        w, b = params[-1]
        return (h @ w + b).squeeze(-1)

    def _clip_grads(tree, max_norm=1.0):
        leaves = jax.tree_util.tree_leaves(tree)
        sq = sum(jp.sum(g * g) for g in leaves)
        scale = jp.minimum(1.0, max_norm / jp.sqrt(sq + 1e-8))
        return jax.tree_util.tree_map(lambda g: jp.nan_to_num(g * scale, nan=0.0), tree)

    def mb_loss(actor, critic, log_std, o_mb, raw_mb, lp_mb, adv_mb, ret_mb):
        mean = mlp_forward(jp, actor, o_mb)
        std = jp.clip(jp.exp(log_std), 1e-3, 1.0)
        new_lp = -0.5 * (((raw_mb - mean) / std) ** 2 + 2.0 * jp.log(std) + jp.log(2.0 * jp.pi)).sum(-1)
        ratio = jp.exp(jp.clip(new_lp - lp_mb, -20.0, 20.0))
        surr1 = ratio * adv_mb
        surr2 = jp.clip(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_mb
        pol = -jp.minimum(surr1, surr2).mean()
        vf = 0.5 * ((critic_apply(critic, o_mb) - ret_mb) ** 2).mean()
        ent = 0.5 * (1.0 + jp.log(2.0 * jp.pi) + 2.0 * jp.log(std)).sum(-1).mean()
        return pol + vf - ENTROPY_COEF * ent, new_lp

    def loss_only(actor, critic, log_std, o_mb, raw_mb, lp_mb, adv_mb, ret_mb):
        loss, _ = mb_loss(actor, critic, log_std, o_mb, raw_mb, lp_mb, adv_mb, ret_mb)
        return loss

    def collect(key, obs, state, actor, critic, log_std):
        def body(carry, _):
            key, obs, state = carry
            key, k = jax.random.split(key)
            mean = mlp_forward(jp, actor, obs)
            std = jp.clip(jp.exp(log_std), 1e-3, 1.0)
            eps = jax.random.normal(k, mean.shape)
            raw = mean + std * eps
            logp = -0.5 * (((raw - mean) / std) ** 2 + 2.0 * jp.log(std) + jp.log(2.0 * jp.pi)).sum(-1)
            v = critic_apply(critic, obs)
            act = jp.tanh(raw)
            out = step_v(*state, act)
            nxt, rew, done = out[-3], out[-2], out[-1]
            new_state = out[:-3]
            return (key, nxt, new_state), (obs, raw, logp, v, rew, done)

        (key, obs, state), traj = jax.lax.scan(body, (key, obs, state), None, length=unroll)
        return key, obs, state, traj

    def gae(rew, done, v, last_v):
        def body(carry, xs):
            last_gae, nxt_v = carry
            r, d, vt = xs
            mask = 1.0 - d
            delta = r + PPO_GAMMA * nxt_v * mask - vt
            gae_t = delta + PPO_GAMMA * PPO_LAM * mask * last_gae
            return (gae_t, vt), gae_t

        zeros = jp.zeros((n_envs,), dtype=rew.dtype)
        _, adv = jax.lax.scan(body, (zeros, last_v), (rew, done, v), reverse=True)
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return jp.nan_to_num(adv, nan=0.0), jp.nan_to_num(ret, nan=0.0)

    def ppo_train(actor, critic, log_std, o_f, raw_f, lp_f, adv_f, ret_f, key):
        def epoch(carry, _):
            actor, critic, log_std, key, stop = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, n_samples)

            def one_mb(carry, i):
                actor, critic, log_std, stop = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                o_mb, raw_mb, lp_mb = o_f[idx], raw_f[idx], lp_f[idx]
                adv_mb, ret_mb = adv_f[idx], ret_f[idx]
                _, new_lp = mb_loss(actor, critic, log_std, o_mb, raw_mb, lp_mb, adv_mb, ret_mb)
                kl = (lp_mb - new_lp).mean()
                stop = stop | (kl > 1.5 * PPO_TARGET_KL)

                def do_upd(_):
                    grads = jax.grad(loss_only, argnums=(0, 1, 2))(
                        actor, critic, log_std, o_mb, raw_mb, lp_mb, adv_mb, ret_mb
                    )
                    grads = (_clip_grads(grads[0]), _clip_grads(grads[1]), _clip_grads(grads[2]))
                    na = jax.tree_util.tree_map(lambda p, g: p - PPO_LR * g, actor, grads[0])
                    nc = jax.tree_util.tree_map(lambda p, g: p - PPO_LR * g, critic, grads[1])
                    ns = jp.clip(log_std - PPO_LR * grads[2], -5.0, 0.0)
                    return na, nc, ns

                def skip(_):
                    return actor, critic, log_std

                actor, critic, log_std = jax.lax.cond(stop, skip, do_upd, operand=None)
                return (actor, critic, log_std, stop), kl

            (actor, critic, log_std, stop), kls = jax.lax.scan(
                one_mb, (actor, critic, log_std, stop), jp.arange(n_mb)
            )
            return (actor, critic, log_std, key, stop), kls

        (actor, critic, log_std, key, _stop), kls = jax.lax.scan(
            epoch, (actor, critic, log_std, key, jp.bool_(False)), None, length=epochs
        )
        return actor, critic, log_std, key, kls

    collect_jit = jax.jit(collect)
    gae_jit = jax.jit(gae)
    ppo_jit = jax.jit(ppo_train)
    print(
        f"jax scan rollout T={unroll} n={n_envs} ppo n_mb={n_mb} (first iter compiles)",
        flush=True,
    )

    for local in range(1, iters + 1):
        global_it = start_iter + local
        new_stage = curriculum_stage(global_it)
        if new_stage != env.stage:
            env.stage = new_stage
            env._compile()
            step_v = env._step_v
            collect_jit = jax.jit(collect)
            rng = jax.random.PRNGKey(seed + global_it)
            obs = env.reset(rng)
        key, obs, env._state, traj = collect_jit(key, obs, env._state, actor, critic, log_std)
        o, raw, logp, v, rew, done = traj
        rew = jp.nan_to_num(rew, nan=0.0)
        last_v = critic_apply(critic, obs)
        adv, ret = gae_jit(rew, done, v, last_v)
        o_f = o.reshape(-1, OBS_DIM)
        raw_f = raw.reshape(-1, ACT_DIM)
        lp_f = logp.reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)
        actor, critic, log_std, key, kls = ppo_jit(
            actor, critic, log_std, o_f, raw_f, lp_f, adv_f, ret_f, key
        )
        mean_ret = float(np.asarray(rew.sum(0).mean()))
        fall_rate = float(np.asarray(done.mean()))
        updates = int(np.asarray((kls <= 1.5 * PPO_TARGET_KL).sum()))
        sps = (n_envs * unroll) / max(time.time() - t0, 1e-6)
        if local == 1 or local % 5 == 0 or local == iters:
            print(
                f"iter {global_it} (+{local}/{iters}) stage={env.stage} ret={mean_ret:.2f} "
                f"fall_rate={fall_rate:.3f} env-steps/s={sps:.0f} elapsed={time.time() - t0:.0f}s jax",
                flush=True,
            )
            print(f"  ppo updates={updates} mb={mb} epochs<={epochs}", flush=True)
        t0 = time.time()

        if local % RESUME_EVERY == 0 or local == iters:
            policy.load_state_dict(jax_params_to_state_dict(actor))
            save_training_checkpoint(
                latest_path_for(out_path),
                actor_sd=policy.state_dict(),
                critic_jax=critic,
                log_std=np.array(log_std, copy=True),
                backend="jax",
                global_iter=global_it,
            )
        if local == iters or local % EVAL_EVERY == 0:
            policy.load_state_dict(jax_params_to_state_dict(actor))
            save_if_stand(policy, out_path, torch_device)
    policy.load_state_dict(jax_params_to_state_dict(actor))
    return policy


def try_mjx(n_envs: int) -> bool:
    err = jax_import_error()
    if err is not None:
        print(f"MJX unavailable ({err}); using CPU MuJoCo vec env", flush=True)
        return False
    try:
        from agent.l3_mjx import MjxFoundationEnv

        env = MjxFoundationEnv(min(4, n_envs), stage=STAGE_STAND)
        import jax

        obs = env.reset(jax.random.PRNGKey(0))
        act = jax.numpy.zeros((env.n, ACT_DIM))
        env.step(act)
        _ = obs
        print(f"MJX ok n={env.n} put_model+step", flush=True)
        return True
    except Exception as exc:
        print(f"MJX unavailable ({exc!r}); using CPU MuJoCo vec env", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=None, help="PPO iterations. Default 200 on JAX GPU, 0 otherwise.")
    p.add_argument("--envs", type=int, default=JAX_DEFAULT_ENVS)
    p.add_argument("--unroll", type=int, default=64)
    p.add_argument("--device", default=os.getenv("L2_DEVICE", "cpu"))
    p.add_argument("--out", default=str(AGENT_ROOT / "flywheel_data" / "l3_foundation.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Resume from l3_foundation.latest.pt (default) or the given checkpoint.",
    )
    p.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Override global_iter after resume (for old checkpoints without the field).",
    )
    args = p.parse_args()
    device = resolve_device(args.device)
    out_path = Path(args.out)
    kind = jax_device_kind()
    mjx_err = jax_import_error()
    gpu_jax = mjx_err is None and not kind.startswith("none") and "cpu" not in kind.lower()
    if mjx_err is not None:
        print(f"jax/mjx import: {mjx_err}", flush=True)
    iters = args.iters if args.iters is not None else (200 if gpu_jax else 0)
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = latest_path_for(out_path) if args.resume == "" else Path(args.resume)
    print(
        f"foundation PPO obs={OBS_DIM} act={ACT_DIM} torch={device} jax={kind} "
        f"iters={iters} envs={args.envs} curriculum_horizon={CURRICULUM_HORIZON} "
        f"stand_only={STAND_ONLY} train_tilt={TRAIN_TILT} prior_scale={BALANCE_PRIOR_SCALE} "
        f"hang_arms=1 contact_term=1 reach_frac={REACH_FRAC} squat_frac={SQUAT_FRAC} "
        f"push={PUSH_FORCE[0]:.0f}-{PUSH_FORCE[1]:.0f}N "
        f"ppo_epochs={PPO_EPOCHS} jax_mb={JAX_MINIBATCH}"
        + (f" resume={resume_path}" if resume_path else ""),
        flush=True,
    )
    net = ActorCritic().to(device)
    resume: dict | None = None
    start_iter = 0
    if resume_path is not None:
        resume = load_training_checkpoint(resume_path, device)
        if resume is None:
            print(f"resume failed: missing or invalid checkpoint {resume_path}", flush=True)
            sys.exit(1)
        loaded = apply_resume_to_net(net, resume)
        start_iter = int(resume.get("global_iter", 0))
        if args.start_iter is not None:
            start_iter = int(args.start_iter)
        print(
            f"resumed {loaded} from {resume_path} global_iter={start_iter} "
            f"next_stage={curriculum_stage(start_iter + 1)}",
            flush=True,
        )
    elif args.start_iter is not None:
        start_iter = int(args.start_iter)
    else:
        # Zero-init last layer + default_q + PD should already stand; save that first.
        save_if_stand(net.actor, out_path, device)
    if iters <= 0:
        return
    if try_mjx(int(args.envs)):
        ppo_jax(
            n_envs=int(args.envs),
            iters=iters,
            unroll=int(args.unroll),
            seed=int(args.seed),
            out_path=out_path,
            torch_device=device,
            resume=resume,
            start_iter=start_iter,
        )
        return
    n_cpu = int(args.envs) if gpu_jax else min(int(args.envs), 8)
    print(f"CPU/PyTorch vec envs n={n_cpu}", flush=True)
    envs = VecFoundationEnv(n_cpu, stage=curriculum_stage(start_iter + 1))
    ppo_torch(
        net=net,
        envs=envs,
        iters=iters,
        unroll=int(args.unroll),
        device=device,
        lr=PPO_LR,
        gamma=PPO_GAMMA,
        lam=PPO_LAM,
        clip=PPO_CLIP,
        epochs=PPO_EPOCHS,
        minibatch=min(256, n_cpu * int(args.unroll)),
        target_kl=PPO_TARGET_KL,
        out_path=out_path,
        start_iter=start_iter,
    )


if __name__ == "__main__":
    main()
