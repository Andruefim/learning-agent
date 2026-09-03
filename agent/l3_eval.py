"""Honest L3 eval suite. Production save gate: reach AND deep squat AND push recovery."""

from __future__ import annotations

import numpy as np

from agent.l3_cmd import CMD_H, CMD_VX, CMD_WZ, reach_command, stand_command
from agent.l3_env import STAGE_STAND, FoundationEnv, _policy_act, load_train_model
from agent.l3_foundation import DECIMATION, WALK_ONLY, squat_cmd_height


def _omega_norm(env: FoundationEnv) -> float:
    return float(np.linalg.norm(env._omega()))


def _nonfoot_ground(env: FoundationEnv) -> bool:
    feet = set(env.r_geoms) | set(env.l_geoms)
    for i in range(int(env.data.ncon)):
        c = env.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in feet or g2 in feet:
            continue
        z1 = float(env.data.geom_xpos[g1, 2])
        z2 = float(env.data.geom_xpos[g2, 2])
        if min(z1, z2) < 0.08:
            return True
    return False


def _pelvis_x(env: FoundationEnv) -> float:
    return float(env.data.xpos[env.pelvis_id, 0])


def _rollout(policy, env: FoundationEnv, obs, *, ticks: int, device, cmd_fn=None):
    z_hist, tilt_hist, pitch_hist = [], [], []
    nonfoot = False
    done = False
    pdt = env.dt * DECIMATION
    x0 = _pelvis_x(env)
    for t in range(max(1, ticks)):
        if cmd_fn is not None:
            env.cmd = cmd_fn(t * pdt)
        obs, _, done = env.step(_policy_act(policy, obs, device))
        z_hist.append(env._z())
        tilt_hist.append(env._tilt())
        pitch_hist.append(float(env._omega()[1]))
        nonfoot = nonfoot or _nonfoot_ground(env)
        if done:
            break
    return {
        "seconds": float(len(z_hist) * pdt),
        "z_min": float(min(z_hist) if z_hist else 0.0),
        "z_last": float(z_hist[-1] if z_hist else 0.0),
        "tilt_min": float(min(tilt_hist) if tilt_hist else 0.0),
        "ticks": int(len(z_hist)),
        "need": int(ticks),
        "fell": bool(done),
        "full": len(z_hist) >= ticks,
        "pitch": pitch_hist,
        "nonfoot": bool(nonfoot),
        "omega": _omega_norm(env),
        "x_delta": _pelvis_x(env) - x0,
    }


def eval_reach(policy, *, device, model, seconds: float = 15.0) -> dict:
    env = FoundationEnv(model, stage=STAGE_STAND)
    obs = env.reset(reach_command(pitch=1.0))
    ticks = int(round(seconds / env._policy_dt()))
    raw = _rollout(policy, env, obs, ticks=ticks, device=device)
    ok = (not raw["fell"]) and raw["full"] and raw["z_min"] >= 0.98 and raw["tilt_min"] >= 0.98
    return {**raw, "ok": bool(ok), "name": "reach"}


def eval_deep_squat(policy, *, device, model) -> dict:
    env = FoundationEnv(model, stage=STAGE_STAND)
    obs = env.reset(stand_command())
    down, hold, up = 1.5, 5.0, 1.5
    total = down + hold + up + 1.0
    ticks = int(round(total / env._policy_dt()))

    def cmd_at(t: float):
        cmd = stand_command()
        cmd[CMD_H] = squat_cmd_height(t)
        return cmd

    raw = _rollout(policy, env, obs, ticks=ticks, device=device, cmd_fn=cmd_at)
    ok = (not raw["fell"]) and raw["full"] and raw["z_min"] <= 0.72 and raw["tilt_min"] >= 0.97
    return {**raw, "ok": bool(ok), "name": "deep_squat"}


def _push_trial(policy, *, device, model, cmd, fx: float, fy: float) -> dict:
    env = FoundationEnv(model, stage=STAGE_STAND)
    obs = env.reset(np.asarray(cmd, dtype=np.float32))
    settle = int(round(0.5 / env._policy_dt()))
    raw0 = _rollout(policy, env, obs, ticks=settle, device=device)
    if raw0["fell"]:
        return {**raw0, "ok": False, "settled": False}
    env.apply_impulse(fx, fy, duration_sec=0.20)
    obs = env._obs()
    rec = int(round(1.5 / env._policy_dt()))
    raw = _rollout(policy, env, obs, ticks=rec, device=device)
    settled = (not raw["fell"]) and raw["full"] and raw["omega"] < 0.05
    return {**raw, "ok": bool(settled), "settled": bool(settled)}


def eval_push_recovery(policy, *, device, model) -> dict:
    cmds = (("stand", stand_command()), ("reach", reach_command(pitch=1.0)))
    dirs = (("front", 50.0, 0.0), ("back", -50.0, 0.0), ("side", 0.0, 50.0))
    trials = []
    all_ok = True
    z_min = 10.0
    tilt_min = 10.0
    fell = False
    for cname, cmd in cmds:
        for dname, fx, fy in dirs:
            r = _push_trial(policy, device=device, model=model, cmd=cmd, fx=fx, fy=fy)
            trials.append(
                {"pose": cname, "dir": dname, **{k: r[k] for k in ("ok", "fell", "omega", "z_min", "tilt_min")}}
            )
            all_ok = all_ok and r["ok"]
            z_min = min(z_min, r["z_min"])
            tilt_min = min(tilt_min, r["tilt_min"])
            fell = fell or r["fell"]
    return {
        "ok": bool(all_ok),
        "name": "push_recovery",
        "z_min": float(z_min),
        "tilt_min": float(tilt_min),
        "fell": bool(fell),
        "seconds": 1.5,
        "trials": trials,
    }


def eval_locomotion(policy, *, device, model, seconds: float = 10.0) -> dict:
    if WALK_ONLY:
        cases = (("fwd", {CMD_VX: 0.30}),)
    else:
        cases = (("fwd", {CMD_VX: 0.5}), ("back", {CMD_VX: -0.3}), ("yaw", {CMD_WZ: 0.5}))
    reports = []
    all_ok = True
    z_min = 10.0
    tilt_min = 10.0
    fell = False
    x_delta = 0.0
    z_last = 0.0
    for name, fields in cases:
        env = FoundationEnv(model, stage=STAGE_STAND)
        cmd = stand_command()
        for k, v in fields.items():
            cmd[k] = float(v)
        obs = env.reset(cmd)
        ticks = int(round(seconds / env._policy_dt()))
        raw = _rollout(policy, env, obs, ticks=ticks, device=device)
        progressed = True
        if name == "fwd":
            progressed = float(raw.get("x_delta", 0.0)) > 0.4
        elif name == "back":
            progressed = float(raw.get("x_delta", 0.0)) < -0.2
        ok = (not raw["fell"]) and raw["full"] and (not raw["nonfoot"]) and progressed
        reports.append(
            {
                "name": name,
                "ok": ok,
                "fell": raw["fell"],
                "nonfoot": raw["nonfoot"],
                "z_min": raw["z_min"],
                "x_delta": raw.get("x_delta", 0.0),
            }
        )
        all_ok = all_ok and ok
        z_min = min(z_min, raw["z_min"])
        tilt_min = min(tilt_min, raw["tilt_min"])
        fell = fell or raw["fell"]
        x_delta = float(raw.get("x_delta", 0.0))
        z_last = float(raw.get("z_last", raw["z_min"]))
    return {
        "ok": bool(all_ok),
        "name": "locomotion",
        "z_min": float(z_min),
        "z_last": float(z_last),
        "tilt_min": float(tilt_min),
        "fell": bool(fell),
        "seconds": float(seconds),
        "x_delta": float(x_delta),
        "trials": reports,
    }


def eval_static_60s(policy, *, device, model, seconds: float = 60.0) -> dict:
    env = FoundationEnv(model, stage=STAGE_STAND)
    obs = env.reset(stand_command())
    ticks = int(round(seconds / env._policy_dt()))
    raw = _rollout(policy, env, obs, ticks=ticks, device=device)
    pitch = np.asarray(raw["pitch"], dtype=np.float64)
    growing = False
    if pitch.size >= 8:
        n = pitch.size
        a0 = float(np.max(np.abs(pitch[: n // 4])))
        a1 = float(np.max(np.abs(pitch[3 * n // 4 :])))
        growing = a1 > a0 * 1.25 + 0.02
    still = float(np.max(np.abs(pitch[-max(1, int(0.5 / env._policy_dt())) :]))) < 0.15 if pitch.size else False
    ok = (not raw["fell"]) and raw["full"] and (not growing) and still
    return {**raw, "ok": bool(ok), "name": "static_60s", "growing": bool(growing)}


def eval_suite(policy, *, device=None, static_sec: float = 60.0) -> dict:
    model = load_train_model()
    if WALK_ONLY:
        loco = eval_locomotion(policy, device=device, model=model)
        static = eval_static_60s(policy, device=device, model=model, seconds=min(4.0, static_sec))
        return {
            "ok": False,
            "seconds": loco["seconds"],
            "z_min": loco["z_min"],
            "tilt_min": loco["tilt_min"],
            "ticks": loco.get("ticks", 0),
            "need": loco.get("need", 0),
            "cases": {"locomotion": loco, "static_60s": static},
        }
    reach = eval_reach(policy, device=device, model=model)
    squat = eval_deep_squat(policy, device=device, model=model)
    push = eval_push_recovery(policy, device=device, model=model)
    loco = eval_locomotion(policy, device=device, model=model)
    static = eval_static_60s(policy, device=device, model=model, seconds=static_sec)
    gate = bool(reach["ok"] and squat["ok"] and push["ok"])
    cases = {
        "reach": reach,
        "deep_squat": squat,
        "push_recovery": push,
        "locomotion": loco,
        "static_60s": static,
    }
    return {
        "ok": gate,
        "seconds": reach["seconds"],
        "z_min": reach["z_min"],
        "tilt_min": reach["tilt_min"],
        "ticks": reach.get("ticks", 0),
        "need": reach.get("need", 0),
        "cases": cases,
    }
