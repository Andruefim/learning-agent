"""PPO trainer for Level3BalancePolicy on Unitree H2 (software Joint-PD).

Run:  .venv/Scripts/python train_l3.py
Saves flywheel_data/l3_balance.pt
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

import mujoco

from agent.config import ROOT as AGENT_ROOT, STAND_Z
from agent.h2 import (
    MODEL_XML,
    SPAWN_Z,
    STAND_Q,
    actuator_addrs,
    disable_foot_spheres,
    joint_limits,
)
from agent.joint_pd import compute_torques, kp_kd_vectors
from agent.l3_controller import (
    ACT_DIM,
    L3_DELTA_LIM,
    LEG_IDX,
    OBS_DIM,
    UPPER_IDX,
    Level3BalancePolicy,
    apply_leg_deltas,
    build_obs,
    legs_nominal,
)

TILT_LIM = float(np.cos(np.deg2rad(35.0)))
FALL_Z = 0.40
PUSH_N = 20.0
ARM_PERIOD = 1.5


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Level3BalancePolicy(zero_out=True)
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.5))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class BalanceEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = self.model.body("pelvis").id
        self.torso_id = self.model.body("torso_link").id
        self.r_foot_id = self.model.body("right_ankle_pitch_link").id
        self.l_foot_id = self.model.body("left_ankle_pitch_link").id
        disable_foot_spheres(self.model, (self.r_foot_id, self.l_foot_id))
        self.qadr, self.vadr = actuator_addrs(self.model)
        self.lo, self.hi = joint_limits(self.model)
        self.kp, self.kd = kp_kd_vectors()
        self.dt = float(self.model.opt.timestep)
        self.q_cmd = STAND_Q.copy()
        self.z_target = STAND_Z
        self.vx_target = 0.0
        self.q_arm_goal = np.zeros(len(UPPER_IDX), dtype=np.float32)
        self._arm_t = 0.0
        self._push_left = 0
        self._rng = np.random.default_rng()

    def _hinges(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float32)

    def _qd(self) -> np.ndarray:
        return self.data.qvel[self.vadr].astype(np.float32)

    def _tilt_up(self) -> float:
        return float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])

    def _sample_arms(self) -> None:
        goal = np.zeros(len(UPPER_IDX), dtype=np.float32)
        kind = int(self._rng.integers(0, 5))
        if kind == 1:
            goal[0] = goal[7] = float(self._rng.uniform(-0.9, -0.3))
        elif kind == 2:
            goal[1] = float(self._rng.uniform(0.25, 0.7))
            goal[8] = float(self._rng.uniform(-0.7, -0.25))
        elif kind == 3:
            goal[0] = float(self._rng.uniform(-0.9, -0.25))
            goal[3] = float(self._rng.uniform(0.15, 0.7))
        elif kind == 4:
            goal[0] = goal[7] = float(self._rng.uniform(-0.5, -0.15))
        self.q_arm_goal = goal

    def _slew_arms(self) -> None:
        cur = self.q_cmd[list(UPPER_IDX)]
        d = np.clip(self.q_arm_goal - cur, -0.015, 0.015)
        self.q_cmd[list(UPPER_IDX)] = np.clip(cur + d, self.lo[list(UPPER_IDX)], self.hi[list(UPPER_IDX)])

    def _hinges(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float32)

    def _qd(self) -> np.ndarray:
        return self.data.qvel[self.vadr].astype(np.float32)

    def _tilt_up(self) -> float:
        return float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])

    def _sample_arms(self) -> None:
        q = self.q_cmd
        q[list(UPPER_IDX)] = 0.0
        kind = int(self._rng.integers(0, 5))
        if kind == 1:
            q[15] = q[22] = float(self._rng.uniform(-1.55, -0.6))
        elif kind == 2:
            q[16] = float(self._rng.uniform(0.4, 1.4))
            q[23] = float(self._rng.uniform(-1.4, -0.4))
        elif kind == 3:
            q[15] = float(self._rng.uniform(-1.55, -0.4))
            q[22] = 0.0
            q[18] = float(self._rng.uniform(0.2, 1.2))
        elif kind == 4:
            q[15] = q[22] = float(self._rng.uniform(-0.8, -0.2))
            q[16] = float(self._rng.uniform(0.2, 0.9))
            q[23] = float(self._rng.uniform(-0.9, -0.2))
        self.q_cmd = np.clip(q, self.lo, self.hi)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.z_target = float(self._rng.uniform(0.80, 0.98))
        h01 = float(np.clip((self.z_target - 0.55) / max(STAND_Z - 0.55, 1e-3), 0.0, 1.0))
        self.q_cmd = legs_nominal(h01)
        self.q_cmd[list(UPPER_IDX)] = 0.0
        self.q_cmd = np.clip(self.q_cmd, self.lo, self.hi)
        self.data.qpos[0:3] = (0.0, 0.0, self.z_target + 0.04)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qadr] = self.q_cmd
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._arm_t = 0.0
        self._push_left = 0
        self.vx_target = 0.0
        self.q_arm_goal = np.zeros(len(UPPER_IDX), dtype=np.float32)
        return self._obs()

    def _obs(self) -> np.ndarray:
        return build_obs(
            self.data,
            self.torso_id,
            self._hinges(),
            self._qd(),
            self.q_cmd,
            self.z_target,
            self.vx_target,
        )

    def _foot_v2(self, body_id: int) -> float:
        v = np.asarray(self.data.cvel[body_id, 3:6], dtype=np.float64)
        return float(np.dot(v, v))

    def _terminated(self) -> bool:
        z = float(self.data.xpos[self.pelvis_id, 2])
        return z < FALL_Z or self._tilt_up() < TILT_LIM

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        self._arm_t += self.dt
        if self._arm_t >= ARM_PERIOD:
            self._arm_t = 0.0
            self._sample_arms()
        self._slew_arms()
        if self._push_left <= 0 and self._rng.random() < 0.004:
            f = self._rng.uniform(-PUSH_N, PUSH_N, size=2)
            self.data.xfrc_applied[self.torso_id, 0:2] = f
            self._push_left = int(self._rng.integers(8, 24))
        elif self._push_left > 0:
            self._push_left -= 1
            if self._push_left <= 0:
                self.data.xfrc_applied[self.torso_id] = 0

        delta = np.clip(np.asarray(action, dtype=np.float32), -L3_DELTA_LIM, L3_DELTA_LIM)
        h01 = float(np.clip((self.z_target - 0.55) / max(STAND_Z - 0.55, 1e-3), 0.0, 1.0))
        q_tgt = apply_leg_deltas(self.q_cmd, delta, height_01=h01)
        q_tgt[list(UPPER_IDX)] = self.q_cmd[list(UPPER_IDX)]
        q_tgt = np.clip(q_tgt, self.lo, self.hi)
        tau = compute_torques(self.model, self.data, q_tgt, self.kp, self.kd, self.qadr, self.vadr)
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        gz = self._tilt_up()
        z = float(self.data.xpos[self.pelvis_id, 2])
        r_up = float(np.exp(-5.0 * (1.0 - gz * gz)))
        r_h = float(np.exp(-10.0 * abs(z - self.z_target)))
        r_feet = -0.5 * (self._foot_v2(self.l_foot_id) + self._foot_v2(self.r_foot_id))
        r_feet = float(np.clip(r_feet, -5.0, 0.0))
        r_tau = -1e-6 * float(np.dot(tau, tau))
        r_tau = float(np.clip(r_tau, -2.0, 0.0))
        reward = r_up + r_h + r_feet + r_tau + 1.0
        done = self._terminated()
        if done:
            reward -= 8.0
        return self._obs(), float(reward), bool(done)


def ppo(
    *,
    iters: int,
    steps_per_iter: int,
    device: torch.device,
    out_path: Path,
    lr: float,
    gamma: float,
    lam: float,
    clip: float,
    epochs: int,
    minibatch: int,
) -> None:
    env = BalanceEnv()
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = env.reset()
    t0 = time.time()
    best = -1e9
    for it in range(1, iters + 1):
        buf_o, buf_a, buf_lp, buf_v, buf_r, buf_d = [], [], [], [], [], []
        ep_ret, ep_len, rets, lens = 0.0, 0, [], []
        for _ in range(steps_per_iter):
            o_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = net.dist(o_t)
                raw = dist.sample()
                logp = dist.log_prob(raw).sum(-1)
                val = net.value(o_t)
            act = torch.tanh(raw) * L3_DELTA_LIM
            nxt, rew, done = env.step(act.squeeze(0).cpu().numpy())
            buf_o.append(obs)
            buf_a.append(raw.squeeze(0).cpu().numpy())
            buf_lp.append(float(logp.cpu()))
            buf_v.append(float(val.cpu()))
            buf_r.append(rew)
            buf_d.append(float(done))
            ep_ret += rew
            ep_len += 1
            obs = nxt
            if done or ep_len >= 2000:
                rets.append(ep_ret)
                lens.append(ep_len)
                ep_ret, ep_len = 0.0, 0
                obs = env.reset()
        adv = np.zeros(steps_per_iter, dtype=np.float32)
        last_gae = 0.0
        with torch.no_grad():
            last_v = float(net.value(torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)).cpu())
        for t in reversed(range(steps_per_iter)):
            nxt_v = last_v if t == steps_per_iter - 1 else buf_v[t + 1]
            mask = 1.0 - buf_d[t]
            delta = buf_r[t] + gamma * nxt_v * mask - buf_v[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + np.asarray(buf_v, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        o = torch.as_tensor(np.stack(buf_o), device=device, dtype=torch.float32)
        a = torch.as_tensor(np.stack(buf_a), device=device, dtype=torch.float32)
        old_lp = torch.as_tensor(buf_lp, device=device, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, device=device, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, device=device, dtype=torch.float32)
        idx = np.arange(steps_per_iter)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, steps_per_iter, minibatch):
                mb = idx[s : s + minibatch]
                dist = net.dist(o[mb])
                logp = dist.log_prob(a[mb]).sum(-1)
                ratio = (logp - old_lp[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t[mb]
                pol = -torch.min(surr1, surr2).mean()
                vf = 0.5 * (net.value(o[mb]) - ret_t[mb]).pow(2).mean()
                ent = -0.01 * dist.entropy().sum(-1).mean()
                loss = pol + vf + ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        mean_ret = float(np.mean(rets)) if rets else float(ep_ret)
        if mean_ret > best:
            best = mean_ret
            if mean_ret > 0:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(net.actor.state_dict(), out_path)
        if it == 1 or it % 10 == 0 or it == iters:
            print(
                f"iter {it}/{iters} ret={mean_ret:.2f} best={best:.2f} "
                f"len={float(np.mean(lens)) if lens else ep_len:.0f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if best > 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.actor.state_dict(), out_path)
        print(f"saved {out_path}", flush=True)
    else:
        print(f"skip save (best ret {best:.2f} <= 0)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--device", default=os.getenv("L2_DEVICE", "cpu"))
    p.add_argument("--out", default=str(AGENT_ROOT / "flywheel_data" / "l3_balance.pt"))
    args = p.parse_args()
    spec = args.device.lower().strip()
    if spec in {"cuda", "gpu", "rocm", "hip"} and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"PPO L3 on {device}, obs={OBS_DIM} act={ACT_DIM}", flush=True)
    ppo(
        iters=int(args.iters),
        steps_per_iter=int(args.steps),
        device=device,
        out_path=Path(args.out),
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        epochs=4,
        minibatch=256,
    )


if __name__ == "__main__":
    main()
