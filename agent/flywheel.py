"""Shadow residual, replay, in-context trials, offline CFM / H1."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import torch

from agent.config import (
    ALPHA_MAX,
    ALPHA_START,
    ALPHA_STEP,
    CHUNK,
    ERROR_LEN,
    H1_SPEC,
    INSTR_BYTES,
    PARAM_KEYS,
    REPLAY_GATE,
    RESIDUAL_LIMIT,
    ROLLBACK_FALL_DELTA,
    SHADOW_EMA,
    SHADOW_EMA_TICKS,
    SHADOW_MSE_MAX,
    SKILL_IDS,
    SKILL_TO_I,
    TRIAL_MAX,
    TURN_IN_CONTEXT_OK,
    VISION_H,
    VISION_STRIDE,
    VISION_W,
    Z_DIM,
)
from agent.h2 import ACTION_DIM, ARM_RAISE, L_HY, L_KN, L_SH, N_ACT, R_HY, R_KN, R_SH, STAND_Q, SQUAT_Q, TRIAL_FEAT
from agent.l3_cmd import clip_command
from agent.plan import Plan, evaluate_trial, plan_to_params
from agent.policy import FlowPolicy, encode_instr
from agent.trials import Trial


class FlywheelMixin:
    def _obs_tensors(self, image, proprio, language, z, errors):
        ids, feat = self.trials.tensors(self.device)
        img = self._images_to_tensor(image if image.ndim == 4 else image[None])
        b = img.shape[0]
        if b != ids.shape[0]:
            ids = ids.expand(b, -1)
            feat = feat.expand(b, -1, -1)
        return (
            img,
            torch.as_tensor(proprio, device=self.device, dtype=torch.float32).reshape(-1, N_ACT),
            torch.as_tensor(language, device=self.device, dtype=torch.long).reshape(-1, INSTR_BYTES),
            torch.as_tensor(z, device=self.device, dtype=torch.float32).reshape(-1, Z_DIM),
            torch.as_tensor(errors, device=self.device, dtype=torch.float32).reshape(-1, ERROR_LEN, 3),
            ids,
            feat,
        )

    def _student(self, image, proprio, language, z, errors) -> np.ndarray:
        with torch.no_grad():
            chunk = self.policy.sample(*self._obs_tensors(image, proprio, language, z, errors))
        act = chunk[0, 0].cpu().numpy().astype(np.float32)
        return clip_command(act)

    def _images_to_tensor(self, images: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.ascontiguousarray(images), device=self.device, dtype=torch.float32)
        return x.permute(0, 3, 1, 2) / 255.0

    def _fall_rate(self) -> float:
        if not self._shadow_falls:
            return 0.0
        return float(sum(self._shadow_falls)) / float(len(self._shadow_falls))

    def _replay_count(self) -> int:
        return int(self._replay_n) + len(self.logs)

    def _maybe_update_authority(self, mse: float, fell: bool):
        """Stage A→B on EMA hold + replay size + non-rising fall rate. Not a raw streak."""
        self._shadow_falls.append(1 if fell else 0)
        fall = self._fall_rate()
        if self.shadow_mse_ema is None:
            self.shadow_mse_ema = float(mse)
        else:
            self.shadow_mse_ema = float(SHADOW_EMA) * float(self.shadow_mse_ema) + (1.0 - float(SHADOW_EMA)) * float(mse)
        ema = float(self.shadow_mse_ema)
        if ema < SHADOW_MSE_MAX:
            if self._ema_ok_tick0 is None:
                self._ema_ok_tick0 = int(self._tick)
                self._ema_ok_fall0 = fall
                self.shadow_ok_streak = 0
            elif fall > self._ema_ok_fall0:
                self._ema_ok_tick0 = int(self._tick)
                self._ema_ok_fall0 = fall
                self.shadow_ok_streak = 0
            self.shadow_ok_streak += VISION_STRIDE
            held = int(self.shadow_ok_streak)
        else:
            self._ema_ok_tick0 = None
            self.shadow_ok_streak = 0
            held = 0
        if (
            self.stage == "A"
            and ema < SHADOW_MSE_MAX
            and held >= SHADOW_EMA_TICKS
            and self._replay_count() >= REPLAY_GATE
            and fall <= self._ema_ok_fall0
        ):
            self.stage = "B"
            self.alpha = ALPHA_START
            self.alpha_working = ALPHA_START
            self._working_fall = fall
            return
        if self.stage in {"B", "C"}:
            if fall > self._working_fall + ROLLBACK_FALL_DELTA or ema > SHADOW_MSE_MAX * 1.6:
                self.alpha = float(self.alpha_working)
                if fall > self._working_fall + 2.0 * ROLLBACK_FALL_DELTA:
                    self.stage = "A"
                    self.alpha = 0.0
                    self.alpha_working = 0.0
                return
            if ema < SHADOW_MSE_MAX and held > 0 and held % SHADOW_EMA_TICKS == 0:
                self.alpha_working = float(self.alpha)
                self._working_fall = fall
                self.alpha = float(min(ALPHA_MAX, self.alpha + ALPHA_STEP))
                if self.alpha >= 0.08:
                    self.stage = "C"

    def _mix(self, teacher: np.ndarray, student: np.ndarray) -> np.ndarray:
        if self.alpha <= 1e-8 or self.outcome == "fall":
            return teacher
        delta = np.clip(student - teacher, -RESIDUAL_LIMIT, RESIDUAL_LIMIT)
        return clip_command(teacher + float(self.alpha) * delta)

    def _open_measure(self):
        p = self._pelvis()
        self._measure = {
            "x0": float(p[0]),
            "yaw0": self._heading(),
            "steps0": int(self._steps_done),
            "ticks": 0,
            "r_sh": [],
            "l_sh": [],
            "r_hy": [],
            "l_hy": [],
        }

    def _poll_measure(self):
        m = self._measure
        if not m:
            return
        m["ticks"] = int(m.get("ticks", 0)) + 1
        h = self._hinges()
        m["r_sh"].append(float(h[R_SH]))
        m["l_sh"].append(float(h[L_SH]))
        m["r_hy"].append(float(h[R_HY]))
        m["l_hy"].append(float(h[L_HY]))

    def _trial_state(self) -> dict:
        m = self._measure or {}
        p = self._pelvis()
        r_sh = m.get("r_sh") or [float(self._hinges()[R_SH])]
        l_sh = m.get("l_sh") or [float(self._hinges()[L_SH])]
        r_hy = m.get("r_hy") or [float(STAND_Q[R_HY])]
        l_hy = m.get("l_hy") or [float(STAND_Q[L_HY])]
        dx = float(p[0]) - float(m.get("x0", p[0]))
        dyaw = self._heading() - float(m.get("yaw0", 0.0))
        dyaw = float(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
        r_arm = float(np.clip(float(np.mean(r_sh[-80:])) / ARM_RAISE, 0.0, 1.0))
        l_arm = float(np.clip(float(np.mean(l_sh[-80:])) / ARM_RAISE, 0.0, 1.0))
        arm_ptp = max(
            float(np.ptp(np.asarray(r_sh[-200:], dtype=np.float32))) if len(r_sh) > 4 else 0.0,
            float(np.ptp(np.asarray(l_sh[-200:], dtype=np.float32))) if len(l_sh) > 4 else 0.0,
        )
        hy = 0.5 * (float(np.mean(r_hy[-40:])) + float(np.mean(l_hy[-40:])))
        span = float(STAND_Q[R_HY] - SQUAT_Q[R_HY])
        h_hat = float(np.clip((hy - float(SQUAT_Q[R_HY])) / max(span, 1e-3), 0.0, 1.2))
        kick_r = float(np.clip((float(STAND_Q[R_HY]) - float(np.min(r_hy))) / 0.32, 0.0, 1.5))
        kick_l = float(np.clip((float(STAND_Q[L_HY]) - float(np.min(l_hy))) / 0.32, 0.0, 1.5))
        return {
            "h": h_hat,
            "vx": float(np.clip(dx / 0.25, -1.0, 1.0)),
            "yaw": self._achieved_yaw() if self._turn_heading0 is not None else float(np.clip(dyaw / 0.9, -1.0, 1.0)),
            "steps": float(max(0, int(self._steps_done) - int(m.get("steps0", 0)))),
            "wave": 1.0 if arm_ptp > 0.18 else 0.0,
            "kick": float(kick_r if kick_r >= kick_l else -kick_l),
            "r_arm": r_arm,
            "l_arm": l_arm,
            "fell": self._fell(),
        }

    def trial_forward(self, skill: str, error: dict, state=None, action=None):
        """One student forward with a synthetic trial token. Used to prove skill_id is in the graph."""
        self.policy.eval()
        raw_s = state if state is not None else np.zeros(N_ACT, dtype=np.float32)
        raw_a = action if action is not None else np.zeros(ACTION_DIM, dtype=np.float32)
        st = (list(map(float, raw_s)) + [0.0] * N_ACT)[:N_ACT]
        act = (list(map(float, raw_a)) + [0.0] * ACTION_DIM)[:ACTION_DIM]
        err = [float(error.get(k, 0.0)) for k in PARAM_KEYS]
        feat = torch.zeros(1, TRIAL_MAX, TRIAL_FEAT, device=self.device)
        ids = torch.full((1, TRIAL_MAX), len(SKILL_IDS), dtype=torch.long, device=self.device)
        ids[0, 0] = int(SKILL_TO_I.get(skill, 0))
        feat[0, 0] = torch.tensor(st[:N_ACT] + act[:ACTION_DIM] + err, device=self.device, dtype=torch.float32)
        image = np.zeros((VISION_H, VISION_W, 3), dtype=np.uint8)
        proprio = np.zeros(N_ACT, dtype=np.float32)
        language = encode_instr(json.dumps({"skill": skill, "params": {}}, sort_keys=True))
        z = np.zeros(Z_DIM, dtype=np.float32)
        z[int(SKILL_TO_I.get(skill, 0))] = 1.0
        errors = np.zeros((ERROR_LEN, 3), dtype=np.float32)
        with torch.no_grad():
            chunk = self.policy.sample(*self._obs_from(image, proprio, language, z, errors, ids, feat))
        return chunk[0, 0].detach().cpu().numpy()

    def trial_token_grad(self, skill: str, error: dict) -> float:
        self.policy.eval()
        err = [float(error.get(k, 0.0)) for k in PARAM_KEYS]
        ids = torch.full((1, TRIAL_MAX), len(SKILL_IDS), dtype=torch.long, device=self.device)
        ids[0, 0] = int(SKILL_TO_I.get(skill, 0))
        row = [0.0] * (N_ACT + ACTION_DIM) + err
        feat = torch.zeros(1, TRIAL_MAX, TRIAL_FEAT, device=self.device, dtype=torch.float32)
        feat[0, 0] = torch.tensor(row, device=self.device, dtype=torch.float32)
        feat = feat.detach().requires_grad_(True)
        tok = self.policy.encode_trials(ids, feat)
        tok.sum().backward()
        return float(feat.grad.abs().sum().item())

    def _obs_from(self, image, proprio, language, z, errors, ids, feat):
        img = self._images_to_tensor(image if getattr(image, "ndim", 3) == 4 else image[None])
        return (
            img,
            torch.as_tensor(proprio, device=self.device, dtype=torch.float32).reshape(-1, N_ACT),
            torch.as_tensor(language, device=self.device, dtype=torch.long).reshape(-1, INSTR_BYTES),
            torch.as_tensor(z, device=self.device, dtype=torch.float32).reshape(-1, Z_DIM),
            torch.as_tensor(errors, device=self.device, dtype=torch.float32).reshape(-1, ERROR_LEN, 3),
            ids,
            feat,
        )

    def _perturb_start(self, skill: str):
        rng = np.random.default_rng()
        if skill == "squat":
            self.data.qpos[self.qadr[R_HY]] += float(rng.uniform(-0.06, 0.06))
            self.data.qpos[self.qadr[L_HY]] += float(rng.uniform(-0.06, 0.06))
            self.data.qpos[self.qadr[R_KN]] += float(rng.uniform(0.0, 0.08))
            self.data.qpos[self.qadr[L_KN]] += float(rng.uniform(0.0, 0.08))
        elif skill == "locomote":
            self.data.qpos[0] += float(rng.uniform(-0.03, 0.03))
            for gid in self._all_foot_geoms:
                self.model.geom_friction[gid, 0] = 0.65 * self._foot_mu0
        mujoco.mj_forward(self.model, self.data)

    def _recover(self, skill: str, fell: bool):
        saved_bias = dict(self._exec_bias)
        self._exec_bias = {}
        try:
            if fell:
                self._home(keep_trials=True, keep_intent=True)
                return
            self.outcome = "ok"
            self.waypoint = Plan.stand(f"recover-{skill}")
            self._steps_goal = 0
            self._step_count = 0
            for _ in range(350):
                self.step()
        finally:
            self._exec_bias = saved_bias

    def _run_trial_chunk(self, plan: Plan, ticks: int) -> Trial:
        self.apply_plan(plan, fresh=True, l1_ok=True)
        self._open_measure()
        goal_steps = int(plan.teacher().steps)
        start_done = int(self._steps_done)
        last_prop = self._hinges().tolist()
        last_act = self._last_teacher.tolist()
        for _ in range(int(ticks)):
            self.step()
            last_prop = self._hinges().tolist()
            last_act = self._last_teacher.tolist()
            if self._fell():
                break
            if goal_steps > 0 and (int(self._steps_done) - start_done) >= goal_steps:
                break
        state = self._trial_state()
        self._measure = None
        params = plan_to_params(plan)
        skill = plan.skill
        success, error_vector = evaluate_trial(skill, params, state)
        achieved = {k: float(state.get(k, 0.0)) for k in PARAM_KEYS}
        trial = Trial(
            skill=skill,
            params=params,
            achieved=achieved,
            error_vector=error_vector,
            fell=bool(state.get("fell", False)),
            success=bool(success),
            state=last_prop,
            action=last_act,
        )
        self.trials.append(trial)
        return trial

    def auto_trial(
        self,
        plan: Plan | None = None,
        max_trials: int = TRIAL_MAX,
        *,
        bias: dict | None = None,
        ticks: int | None = None,
        perturb: bool = False,
    ) -> dict:
        """Execute up to max_trials. Correction is the FlowPolicy forward, not params-error."""
        plan = plan or self.goal or self.waypoint
        if plan is None:
            plan = Plan.stand()
        skill = plan.skill
        if skill == "turn" and not TURN_IN_CONTEXT_OK:
            return {
                "skill": skill,
                "lines": ["Auto-Trial skipped: turn not accepted for in-context yet"],
                "trials": [],
                "success": False,
                "sgd": int(self._sgd_steps),
                "skipped": True,
            }
        if ticks is None:
            ticks = 1400 if skill == "locomote" else 700
        self.policy.eval()
        sgd_before = int(self._sgd_steps)
        self._trial_busy = True
        lines: list[str] = []
        try:
            with self.lock:
                self.trials.clear()
                self._exec_bias = {str(k): float(v) for k, v in (bias or {}).items()}
                self._home(keep_trials=True, keep_intent=True)
                self.goal = plan
                last = None
                for n in range(1, int(max_trials) + 1):
                    if n > 1:
                        self._recover(skill, fell=bool(last and last.fell))
                    if perturb and n == 1:
                        self._perturb_start(skill)
                    last = self._run_trial_chunk(plan, ticks)
                    line = last.log_line(n)
                    lines.append(line)
                    if last.success:
                        break
                    if last.fell:
                        self._home(keep_trials=True, keep_intent=True)
                self._exec_bias = {}
                self._restore_feet()
                self._home(keep_trials=True, keep_intent=False)
                self.status = "auto-trial"
        finally:
            self._trial_busy = False
            self._exec_bias = {}
            self._restore_feet()
        assert self._sgd_steps == sgd_before, "in-context path must not backprop"
        assert not self.policy.training
        return {
            "skill": skill,
            "lines": lines,
            "trials": [t.as_public(i + 1) for i, t in enumerate(self.trials.items())],
            "success": bool(self.trials.items() and self.trials.items()[-1].success),
            "sgd": self._sgd_steps,
        }

    def _replay_compatible(self, blob) -> bool:
        need = ("image", "language", "z", "errors", "action", "proprio")
        if any(k not in blob for k in need):
            return False
        return (
            blob["action"].shape[-1] == ACTION_DIM
            and blob["language"].ndim == 2
            and blob["language"].shape[1] == INSTR_BYTES
            and blob["z"].shape[-1] == Z_DIM
            and blob["errors"].shape[-2:] == (ERROR_LEN, 3)
            and blob["image"].ndim == 4
            and blob["proprio"].shape[-1] == N_ACT
        )

    def _save_replay(self, rows: list[dict]):
        if not rows:
            return
        pack = {
            "image": np.stack([r["image"] for r in rows]).astype(np.uint8),
            "proprio": np.array([r["proprio"] for r in rows], dtype=np.float32),
            "language": np.array([r["language"] for r in rows], dtype=np.int64),
            "z": np.array([r["z"] for r in rows], dtype=np.float32),
            "errors": np.array([r["errors"] for r in rows], dtype=np.float32),
            "action": np.array([r["action"] for r in rows], dtype=np.float32),
        }
        if self.replay_path.exists():
            old = np.load(self.replay_path)
            if self._replay_compatible(old):
                pack = {k: np.concatenate([old[k], pack[k]], axis=0)[-8_000:] for k in pack}
        np.savez_compressed(self.replay_path, **pack)
        self._replay_n = int(pack["action"].shape[0])

    def _load_replay(self) -> dict[str, np.ndarray] | None:
        if not self.replay_path.exists():
            return None
        blob = dict(np.load(self.replay_path))
        if not self._replay_compatible(blob):
            return None
        return blob

    def _fit_cfm(self, policy, optimizer, images, proprio, language, z, errors, actions, steps: int = 100) -> float:
        policy.train()
        last = 0.0
        n = len(actions)
        bs = min(32, n - CHUNK + 1)
        for _ in range(steps):
            starts = torch.randint(0, n - CHUNK + 1, (bs,), device=self.device)
            idx = starts[:, None] + torch.arange(CHUNK, device=self.device)
            loss = policy.cfm_loss(
                images[starts], proprio[starts], language[starts], z[starts], errors[starts], actions[idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._sgd_steps += 1
            last = float(loss.item())
        policy.eval()
        return last

    def _replay_tensors(self, blob: dict, cap: int = 4000):
        images = self._images_to_tensor(blob["image"][-cap:])
        proprio = torch.as_tensor(blob["proprio"][-cap:], device=self.device, dtype=torch.float32)
        language = torch.as_tensor(blob["language"][-cap:], device=self.device, dtype=torch.long)
        z = torch.as_tensor(blob["z"][-cap:], device=self.device, dtype=torch.float32)
        errors = torch.as_tensor(blob["errors"][-cap:], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(blob["action"][-cap:], device=self.device, dtype=torch.float32)
        return images, proprio, language, z, errors, actions

    def evaluate_h1(self) -> dict:
        blob = self._load_replay()
        if blob is None or len(blob["action"]) < 64:
            self.h1_pass = False
            self.student_drive = False
            self.h1_report = {"h1": False, "reason": "need replay", **H1_SPEC}
            return self.h1_report
        images, proprio, language, z, errors, actions = self._replay_tensors(blob)
        n = len(actions)
        split = max(CHUNK + 8, int(n * 0.8))
        ev = slice(split, n)
        if actions[ev].shape[0] < 8:
            ev = slice(max(0, n - 32), n)
        zeros = torch.zeros_like(errors)
        shuffled = errors[torch.randperm(n, device=self.device)]
        abl = FlowPolicy().to(self.device)
        self._fit_cfm(abl, torch.optim.AdamW(abl.parameters(), lr=1e-3), images, proprio, language, z, shuffled, actions)

        def first_action(policy, err):
            with torch.no_grad():
                pred = policy.sample(images[ev], proprio[ev], language[ev], z[ev], err[ev])
            return pred[:, 0]

        teacher = actions[ev]
        mse = lambda a, b: float(torch.mean((a - b) ** 2).item())
        mse_true = mse(first_action(self.policy, errors), teacher)
        mse_zero = mse(first_action(self.policy, zeros), teacher)
        mse_abl = mse(first_action(abl, errors), teacher)
        gap = mse_zero - mse_true
        vs_abl = mse_abl - mse_true
        passed = gap > 1e-4 and vs_abl > 1e-4 and mse_true < 0.05
        self.h1_pass = passed
        self.student_drive = False
        self.h1_report = {
            **H1_SPEC,
            "h1": passed,
            "mse_true": round(mse_true, 6),
            "mse_zero_deque": round(mse_zero, 6),
            "mse_shuffled_train": round(mse_abl, 6),
            "gap_zero": round(gap, 6),
            "gap_abl": round(vs_abl, 6),
            "drive": False,
            "stage": self.stage,
            "alpha": round(float(self.alpha), 3),
            "shadow_mse": round(float(self.shadow_mse), 4),
            "shadow_mse_ema": None if self.shadow_mse_ema is None else round(float(self.shadow_mse_ema), 4),
            "shadow_ok_streak": int(self.shadow_ok_streak),
            "fall_rate": round(self._fall_rate(), 3),
        }
        return self.h1_report

    def consolidate(self) -> str:
        with self.lock:
            session = list(self.logs)
            if len(session) < 32:
                return "Need more motion first — give a command, wait a few seconds, then retry."
            self._save_replay(session)
            blob = self._load_replay()
            if blob is None:
                return "Replay is empty or incompatible (old arm buffer was skipped)."
            n = len(blob["action"])
            if n < CHUNK:
                return f"Need {CHUNK} frames, have {n}."
            tensors = self._replay_tensors(blob)
            last = self._fit_cfm(self.policy, self.optimizer, *tensors)
            torch.save(self.policy.state_dict(), self.ckpt)
            self.baked = True
            self.student_drive = False
            self.logs.clear()
            report = self.evaluate_h1()
            ema = 0.0 if self.shadow_mse_ema is None else float(self.shadow_mse_ema)
            return (
                f"[Consolidate] Replay: {n} | Loss: {last:.5f} | Shadow MSE EMA: {ema:.4f} | "
                f"Fall rate: {self._fall_rate():.3f} | Stage: {self.stage} | Alpha: {self.alpha:.3f}"
                f" · H1={'pass' if report.get('h1') else 'fail'}"
            )

