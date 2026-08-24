"""Playground-style PPO for HumanoidFoundationPolicy.

Prefers MJX-JAX on ROCm (512+ envs). Falls back to in-process MuJoCo vec envs.
Writes flywheel_data/l3_foundation.pt only if eval_reach, eval_deep_squat, and
eval_push_recovery all pass (honest 15s reach / deep squat / push recovery).

Run:  python train_l3_foundation.py
      python train_l3_foundation.py --iters 200 --envs 512
"""

from __future__ import annotations

import argparse
import os
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
    OBS_DIM,
    HumanoidFoundationPolicy,
    jax_params_to_state_dict,
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


def curriculum_stage(it: int, iters: int) -> int:
    if it < max(1, iters // 5):
        return STAGE_STAND
    if it < max(2, (2 * iters) // 5):
        return STAGE_VX
    return STAGE_FULL


RESUME_EVERY = 5  # unconditional checkpoint cadence, independent of the production gate


def latest_path_for(out_path: Path) -> Path:
    """Resume checkpoint next to the gated one. Different name so engine.py
    (which only loads `l3_foundation.pt`) never picks this up by accident."""
    return out_path.with_name(f"{out_path.stem}.latest.pt")


def save_latest(policy: HumanoidFoundationPolicy, path: Path) -> None:
    """Unconditional snapshot for resuming/inspecting training. Not gated,
    not loaded by the live app - only `save_if_stand`'s output is."""
    sd = policy.state_dict()
    if not all(torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v)):
        print(f"resume checkpoint skipped (non-finite weights) {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    print(f"resume checkpoint saved {path}", flush=True)


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
) -> HumanoidFoundationPolicy:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.as_tensor(envs.reset(), device=device, dtype=torch.float32)
    t0 = time.time()
    n = envs.n
    for it in range(1, iters + 1):
        envs.stage = curriculum_stage(it, iters)
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
                ent = -0.01 * dist.entropy().sum(-1).mean()
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
        sps = env_steps / max(time.time() - t0, 1e-6)
        if it == 1 or it % 5 == 0 or it == iters:
            print(
                f"iter {it}/{iters} stage={envs.stage} ret={mean_ret:.2f} "
                f"env-steps/s={sps:.0f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        t0 = time.time()
        if it % RESUME_EVERY == 0 or it == iters:
            save_latest(net.actor, latest_path_for(out_path))
        if it == iters or it % 20 == 0:
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
) -> HumanoidFoundationPolicy:
    import jax
    import jax.numpy as jp

    from agent.l3_mjx import MjxFoundationEnv, mlp_forward, mlp_init

    env = MjxFoundationEnv(n_envs, stage=STAGE_STAND)
    key = jax.random.PRNGKey(seed)
    key, k_p, k_c = jax.random.split(key, 3)
    actor = mlp_init(jax, jp, k_p, zero_out=True)

    def init_critic(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return [
            (jax.random.normal(k1, (OBS_DIM, 256)) * jp.sqrt(2.0 / OBS_DIM), jp.zeros((256,))),
            (jax.random.normal(k2, (256, 256)) * jp.sqrt(2.0 / 256), jp.zeros((256,))),
            (jp.zeros((256, 1)), jp.zeros((1,))),
        ]

    critic = init_critic(k_c)
    log_std = jp.full((ACT_DIM,), -1.2)
    policy = HumanoidFoundationPolicy(zero_out=True)
    rng = jax.random.PRNGKey(seed + 1)
    obs = env.reset(rng)
    t0 = time.time()

    def critic_apply(params, x):
        h = x
        for w, b in params[:-1]:
            z = h @ w + b
            h = z * jax.nn.sigmoid(z)
        w, b = params[-1]
        return (h @ w + b).squeeze(-1)

    for it in range(1, iters + 1):
        new_stage = curriculum_stage(it, iters)
        if new_stage != env.stage:
            env.stage = new_stage
            env._compile()
            rng = jax.random.PRNGKey(seed + it)
            obs = env.reset(rng)
        buf_o, buf_raw, buf_lp, buf_v, buf_r, buf_d = [], [], [], [], [], []
        for _ in range(unroll):
            key, k = jax.random.split(key)
            mean = mlp_forward(jp, actor, obs)
            std = jp.clip(jp.exp(log_std), 1e-3, 1.0)
            eps = jax.random.normal(k, mean.shape)
            raw = mean + std * eps
            logp = -0.5 * (((raw - mean) / std) ** 2 + 2.0 * jp.log(std) + jp.log(2.0 * jp.pi)).sum(-1)
            v = critic_apply(critic, obs)
            act = jp.tanh(raw)
            nxt, rew, done = env.step(act)
            buf_o.append(obs)
            buf_raw.append(raw)
            buf_lp.append(logp)
            buf_v.append(v)
            buf_r.append(rew)
            buf_d.append(done)
            obs = nxt
        o = jp.stack(buf_o)
        raw = jp.stack(buf_raw)
        logp = jp.stack(buf_lp)
        v = jp.stack(buf_v)
        rew = jp.nan_to_num(jp.stack(buf_r), nan=0.0)
        done = jp.stack(buf_d)
        mean_ret = float(np.asarray(rew.sum(0).mean()))
        sps = (n_envs * unroll) / max(time.time() - t0, 1e-6)
        if it == 1 or it % 5 == 0 or it == iters:
            print(
                f"iter {it}/{iters} stage={env.stage} ret={mean_ret:.2f} "
                f"env-steps/s={sps:.0f} elapsed={time.time() - t0:.0f}s jax",
                flush=True,
            )
        t0 = time.time()
        adv = jp.zeros_like(rew)
        last_gae = jp.zeros((n_envs,))
        last_v = critic_apply(critic, obs)
        for t in range(unroll - 1, -1, -1):
            nxt_v = last_v if t == unroll - 1 else v[t + 1]
            mask = 1.0 - done[t]
            delta = rew[t] + 0.99 * nxt_v * mask - v[t]
            last_gae = delta + 0.99 * 0.95 * mask * last_gae
            adv = adv.at[t].set(last_gae)
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = jp.nan_to_num(adv, nan=0.0)
        ret = jp.nan_to_num(ret, nan=0.0)

        def loss_fn(actor, critic, log_std):
            mean = mlp_forward(jp, actor, o)
            std = jp.clip(jp.exp(log_std), 1e-3, 1.0)
            new_lp = -0.5 * (((raw - mean) / std) ** 2 + 2.0 * jp.log(std) + jp.log(2.0 * jp.pi)).sum(-1)
            ratio = jp.exp(jp.clip(new_lp - logp, -20.0, 20.0))
            surr1 = ratio * adv
            surr2 = jp.clip(ratio, 0.8, 1.2) * adv
            pol = -jp.minimum(surr1, surr2).mean()
            vf = 0.5 * ((critic_apply(critic, o) - ret) ** 2).mean()
            return pol + vf

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(actor, critic, log_std)

        def _clip_grads(tree, max_norm=1.0):
            leaves = jax.tree_util.tree_leaves(tree)
            sq = sum(jp.sum(g * g) for g in leaves)
            scale = jp.minimum(1.0, max_norm / jp.sqrt(sq + 1e-8))
            return jax.tree_util.tree_map(lambda g: jp.nan_to_num(g * scale, nan=0.0), tree)

        grads = (_clip_grads(grads[0]), _clip_grads(grads[1]), _clip_grads(grads[2]))
        lr = 3e-4
        actor = jax.tree_util.tree_map(lambda p, g: p - lr * g, actor, grads[0])
        critic = jax.tree_util.tree_map(lambda p, g: p - lr * g, critic, grads[1])
        log_std = jp.clip(log_std - lr * grads[2], -5.0, 0.0)
        if it % RESUME_EVERY == 0 or it == iters:
            policy.load_state_dict(jax_params_to_state_dict(actor))
            save_latest(policy, latest_path_for(out_path))
        if it == iters or it % 20 == 0:
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
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--unroll", type=int, default=64)
    p.add_argument("--device", default=os.getenv("L2_DEVICE", "cpu"))
    p.add_argument("--out", default=str(AGENT_ROOT / "flywheel_data" / "l3_foundation.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = resolve_device(args.device)
    out_path = Path(args.out)
    kind = jax_device_kind()
    mjx_err = jax_import_error()
    gpu_jax = mjx_err is None and not kind.startswith("none") and "cpu" not in kind.lower()
    if mjx_err is not None:
        print(f"jax/mjx import: {mjx_err}", flush=True)
    iters = args.iters if args.iters is not None else (200 if gpu_jax else 0)
    print(
        f"foundation PPO obs={OBS_DIM} act={ACT_DIM} torch={device} jax={kind} "
        f"iters={iters} envs={args.envs}",
        flush=True,
    )
    net = ActorCritic().to(device)
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
        )
        return
    n_cpu = int(args.envs) if gpu_jax else min(int(args.envs), 8)
    print(f"CPU/PyTorch vec envs n={n_cpu}", flush=True)
    envs = VecFoundationEnv(n_cpu, stage=STAGE_STAND)
    ppo_torch(
        net=net,
        envs=envs,
        iters=iters,
        unroll=int(args.unroll),
        device=device,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        epochs=4,
        minibatch=min(256, n_cpu * int(args.unroll)),
        target_kl=0.02,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
