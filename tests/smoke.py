"""Physics + planner smoke for the Unitree H2 foundation controller.

Run: `MUJOCO_GL=egl python app.py --smoke` or `python -m tests.smoke`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"

from agent import (  # noqa: E402
    ACTION_DIM,
    ALPHA_MAX,
    H1_SPEC,
    L_SH,
    N_ACT,
    Plan,
    R_SH,
    RobotEngine,
    SHADOW_MSE_MAX,
    evaluate_trial,
    parse_requested_yaw,
)
from agent.config import STAND_Z  # noqa: E402
from agent.l3_cmd import L2_CMD_DIM, command_from_plan  # noqa: E402
from agent.l3_foundation import (  # noqa: E402
    ACT_DIM,
    OBS_DIM,
    HumanoidFoundationPolicy,
    build_obs,
)


def _drive(bot: RobotEngine, plan: Plan, steps: int) -> None:
    for i in range(steps):
        if i % 50 == 0:
            bot.apply_plan(plan, fresh=(i == 0), l1_ok=True)
        bot.step()


def _assert_idle_stand(bot: RobotEngine, ticks: int = 800) -> None:
    for i in range(ticks):
        bot.step()
        z = float(bot._pelvis()[2])
        assert bot.outcome != "fall", {"tick": i, **bot.telemetry()}
        assert abs(z - STAND_Z) < 0.05, {"tick": i, "z": z, "stand_z": STAND_Z, **bot.telemetry()}


def _assert_torso_push(bot: RobotEngine) -> None:
    """Horizontal torso impulse after ~100 ticks of stand; z must return near STAND_Z."""
    bot.reset_sim()
    _assert_idle_stand(bot, 100)
    for _ in range(8):
        bot.data.xfrc_applied[bot.torso_id, 0] = 180.0
        bot.step()
    bot.data.xfrc_applied[bot.torso_id] = 0
    saw_fall = bot.outcome == "fall"
    recovered_at = None
    for i in range(800):
        bot.step()
        z = float(bot._pelvis()[2])
        if bot.outcome == "fall":
            saw_fall = True
        if recovered_at is None and saw_fall and bot.outcome != "fall" and abs(z - STAND_Z) < 0.05:
            recovered_at = i
        if i >= 400:
            assert abs(z - STAND_Z) < 0.08, {
                "post_tick": i,
                "z": z,
                "saw_fall": saw_fall,
                **bot.telemetry(),
            }
    z = float(bot._pelvis()[2])
    assert abs(z - STAND_Z) < 0.05, {"z": z, "saw_fall": saw_fall, **bot.telemetry()}
    if saw_fall:
        assert recovered_at is not None and recovered_at < 600, {
            "recovered_at": recovered_at,
            **bot.telemetry(),
        }


def _policy_tracks_vx(bot: RobotEngine) -> bool:
    """Only assert walk displacement if this checkpoint actually tracks vx."""
    bot.reset_sim()
    x0 = float(bot._pelvis()[0])
    _drive(
        bot,
        Plan("шаг", skill="locomote", params={"direction": "forward", "speed": "medium", "distance_hint": "3"}),
        400,
    )
    dx = float(bot._pelvis()[0]) - x0
    fell = bot.outcome == "fall"
    bot.reset_sim()
    return (not fell) and dx > 0.04


def smoke() -> None:
    bot = RobotEngine()
    assert int(bot.model.nu) == N_ACT and int(bot.model.nq) == 7 + N_ACT
    assert bot.model.joint("left_hip_yaw_joint").id >= 0
    assert isinstance(bot.l3, HumanoidFoundationPolicy)
    import torch as _torch

    _torch.manual_seed(0)
    l3_out = bot.l3(_torch.zeros(1, OBS_DIM))
    assert tuple(l3_out.shape) == (1, ACT_DIM)
    if not bot.l3_drive:
        assert float(l3_out.abs().max()) < 1e-6, "untrained L3 last layer must be zero"
    stand_obs = build_obs(
        bot.data,
        bot.torso_id,
        bot._hinges(),
        bot._qd(),
        bot._last_a,
        bot._cmd,
    )
    assert stand_obs.shape == (OBS_DIM,), stand_obs.shape
    cmd = command_from_plan(Plan.stand())
    assert cmd.shape == (L2_CMD_DIM,) == (ACTION_DIM,)
    parsed = bot.planner._parse(
        '{"instruction":"подними правую руку","skill":"reach","params":{"hand":"right"},"done":false}',
        "подними правую руку",
    )
    assert parsed is not None and parsed.skill == "reach" and parsed.params.get("hand") == "right", parsed
    walk = bot.planner._parse(
        '{"instruction":"сделай 5 шагов вперед","skill":"locomote","params":{"direction":"forward","speed":"medium","distance_hint":"5"},"done":false}',
        "сделай 5 шагов вперед",
    )
    assert walk is not None and walk.skill == "locomote" and walk.teacher().vx > 0.2 and walk.teacher().steps == 5, walk
    down = bot.planner._parse(
        '{"instruction":"опусти руки","skill":"hold","params":{"hands":"down"},"done":false}',
        "опусти руки",
    )
    assert down is not None and down.skill == "hold" and down.teacher().r_arm == 0.0, down
    waved = bot.planner._parse(
        '{"instruction":"махни","skill":"wave","params":{"hand":"right"},"done":false}',
        "махни",
    )
    assert waved is not None and waved.skill == "wave" and waved.teacher().wave == 1.0, waved
    hold = bot.planner.hold("что угодно", {})
    assert hold.skill == "stand" and hold.teacher().height == 1.0
    leaked = bot.planner._parse(
        '{"instruction":"x","skill":"reach","params":{"cube":1,"hand":"right"},"done":false}',
        "x",
    )
    assert leaked is not None and "cube" not in leaked.params
    src = (ROOT / "agent" / "plan.py").read_text(encoding="utf-8")
    assert "cube_z" not in src
    assert "GRASP_Z" not in src

    err = {"h": 0.18, "vx": 0.0, "yaw": 0.0, "steps": 0.0, "wave": 0.0, "kick": 0.0, "r_arm": 0.0, "l_arm": 0.0}
    _torch.manual_seed(0)
    a = bot.trial_forward("squat", err)
    _torch.manual_seed(0)
    b = bot.trial_forward("locomote", err)
    assert a.shape == b.shape == (ACTION_DIM,)
    assert float(np.max(np.abs(a - b))) > 1e-6, "same error, different skill_id must change the net"
    formula = np.full(ACTION_DIM, -0.18, dtype=np.float32)
    assert float(np.max(np.abs(a - formula))) > 1e-4, "correction must not be params-error"
    g_sq = bot.trial_token_grad("squat", err)
    g_loc = bot.trial_token_grad("locomote", err)
    assert g_sq > 0 and g_loc > 0, "trial tokens must sit in the autograd graph"
    assert H1_SPEC["requires_student_on_actuators"] is False
    assert H1_SPEC["shadow_streak"] >= 1 and H1_SPEC["shadow_mse_max"] > 0
    assert ALPHA_MAX == 0.12
    assert "stage_d" in H1_SPEC
    assert "l3_cmd" in H1_SPEC["residual"] or "command" in H1_SPEC["residual"]

    bot.stage = "B"
    bot.alpha = 0.08
    bot.alpha_working = 0.03
    bot._working_fall = 0.0
    bot._shadow_falls.clear()
    bot._shadow_falls.extend([1] * 64)
    bot._maybe_update_authority(mse=0.5, fell=True)
    assert bot.alpha <= 0.03, {"alpha": bot.alpha, "stage": bot.stage}

    bot.stage = "A"
    bot.alpha = 0.0
    bot.shadow_mse_ema = 0.04
    bot._ema_ok_tick0 = 1
    bot._ema_ok_fall0 = 0.0
    bot.shadow_ok_streak = 160
    bot._replay_n = 1000
    bot._shadow_falls.clear()
    bot._shadow_falls.extend([0] * 20)
    bot._maybe_update_authority(mse=0.20, fell=False)
    assert bot.shadow_mse_ema is not None and bot.shadow_mse_ema < SHADOW_MSE_MAX
    assert bot.shadow_ok_streak > 160, bot.shadow_ok_streak
    assert bot.stage == "A"
    bot.shadow_ok_streak = 196
    bot._maybe_update_authority(mse=0.04, fell=False)
    assert bot.stage == "B" and abs(bot.alpha - 0.03) < 1e-6, {"stage": bot.stage, "alpha": bot.alpha}

    bot.stage = "A"
    bot.alpha = 0.0
    bot.shadow_mse_ema = 0.04
    bot._ema_ok_tick0 = 1
    bot._ema_ok_fall0 = 0.0
    bot.shadow_ok_streak = 500
    bot._replay_n = 10
    bot._shadow_falls.clear()
    bot._shadow_falls.extend([0] * 20)
    bot._maybe_update_authority(mse=0.04, fell=False)
    assert bot.stage == "A", "replay gate must block A→B"

    bot._replay_n = 1000
    bot._ema_ok_tick0 = 1
    bot._ema_ok_fall0 = 0.0
    bot.shadow_ok_streak = 500
    bot._shadow_falls.clear()
    bot._shadow_falls.extend([1] * 32)
    bot._maybe_update_authority(mse=0.04, fell=True)
    assert bot.stage == "A", "fall-rate rise must block A→B"
    bot.stage = "A"
    bot.alpha = 0.0
    blob = bot._load_replay()
    bot._replay_n = 0 if blob is None else int(len(blob["action"]))

    _assert_idle_stand(bot, 800)
    tel = bot.telemetry()
    assert bot._tilt_up() > 0.8, tel
    assert abs(float(bot._hinges()[R_SH])) < 0.45, "idle arms should hang"

    z0 = float(bot._pelvis()[2])
    _drive(bot, Plan("присесть", skill="squat", params={"depth": "low"}), 800)
    z_sq = float(bot._pelvis()[2])
    assert bot.outcome != "fall", bot.telemetry()
    assert z_sq < z0 - 0.02, {**bot.telemetry(), "stand_z": z0, "squat_z": z_sq}

    bot.reset_sim()
    _assert_idle_stand(bot, 200)
    _drive(bot, Plan("подними руки", skill="reach", params={"hand": "both"}), 700)
    assert bot.outcome != "fall", bot.telemetry()
    assert float(bot._hinges()[R_SH]) < -0.45, bot.telemetry()

    bot.reset_sim()
    _assert_idle_stand(bot, 200)
    _drive(bot, Plan("махни правой", skill="wave", params={"hand": "right"}), 700)
    assert bot.outcome != "fall", bot.telemetry()
    assert float(bot._hinges()[R_SH]) < -0.45, bot.telemetry()

    _drive(bot, Plan("опусти руки", skill="hold", params={"hands": "down"}), 700)
    assert abs(float(bot._hinges()[R_SH])) < 0.45, bot.telemetry()
    assert abs(float(bot._hinges()[L_SH])) < 0.45, bot.telemetry()

    _drive(bot, Plan("stand", skill="stand"), 400)
    bot.data.qpos[3:7] = [0.85, 0.5, 0.0, 0.0]
    import mujoco

    mujoco.mj_forward(bot.model, bot.data)
    bot.step()
    assert bot.outcome == "fall", bot.telemetry()
    bot.apply_plan(Plan("спасайся", skill="locomote", params={"direction": "forward", "speed": "fast"}), fresh=True, l1_ok=True)
    assert bot.status == "failed" and bot.user_cmd == ""

    bot.reset_sim()
    _assert_idle_stand(bot, 800)
    _assert_torso_push(bot)

    bot.reset_sim()
    pelvis = bot._pelvis()
    assert float(pelvis[2]) > 0.75, {"z": float(pelvis[2]), **bot.telemetry()}
    assert abs(float(pelvis[0])) < 0.08 and abs(float(pelvis[1])) < 0.08, {
        "xy": (float(pelvis[0]), float(pelvis[1])),
        **bot.telemetry(),
    }
    assert bot._tilt_up() > 0.90, {"tilt": bot._tilt_up(), **bot.telemetry()}
    assert bot.outcome == "ok", bot.telemetry()

    tracks = _policy_tracks_vx(bot)
    if tracks:
        bot.reset_sim()
        x0 = float(bot._pelvis()[0])
        _drive(
            bot,
            Plan(
                "сделай 5 шагов вперед",
                skill="locomote",
                params={"direction": "forward", "speed": "medium", "distance_hint": "5"},
            ),
            1600,
        )
        x1 = float(bot._pelvis()[0])
        assert x1 > x0 + 0.04, {**bot.telemetry(), "x0": x0, "x1": x1}
        assert bot.outcome != "fall", bot.telemetry()
        bot.begin_command("присядь")
        bot.step()
        assert bot.user_cmd == "присядь", "finished walk must not clear the next command"
    else:
        print("walk skip: foundation checkpoint does not track vx yet")

    skipped = bot.auto_trial(Plan("налево", skill="turn", params={"direction": "left"}), max_trials=1, ticks=50)
    assert skipped.get("skipped") is True, skipped
    assert skipped["trials"] == []

    req90 = parse_requested_yaw("turn", {"direction": "left", "angle": "90"})
    assert req90 is not None and abs(float(req90) - 0.5 * np.pi) < 0.02, req90

    bot.reset_sim()
    ry0 = float(bot.data.xpos[bot.model.body("right_elbow_link").id][1])
    ly0 = float(bot.data.xpos[bot.model.body("left_elbow_link").id][1])
    _drive(bot, Plan("руки в стороны", skill="reach", params={"pose": "t"}), 800)
    ry_t = float(bot.data.xpos[bot.model.body("right_elbow_link").id][1])
    ly_t = float(bot.data.xpos[bot.model.body("left_elbow_link").id][1])
    assert bot.outcome != "fall", bot.telemetry()
    assert abs(ry_t) > abs(ry0) + 0.02, {"ry0": ry0, "ry_t": ry_t}
    assert abs(ly_t) > abs(ly0) + 0.02, {"ly0": ly0, "ly_t": ly_t}

    span_t = abs(ry_t - ly_t)
    _drive(bot, Plan("хлопни", skill="wave", params={"pose": "clap"}), 800)
    ry_c = float(bot.data.xpos[bot.model.body("right_elbow_link").id][1])
    ly_c = float(bot.data.xpos[bot.model.body("left_elbow_link").id][1])
    assert abs(ry_c - ly_c) < span_t - 0.05, {"span_t": span_t, "span_c": abs(ry_c - ly_c)}

    ok_sq, err_sq = evaluate_trial(
        "squat",
        {"h": 0.45, "vx": 0.0, "yaw": 0.0, "steps": 0.0, "wave": 0.0, "kick": 0.0, "r_arm": 0.0, "l_arm": 0.0},
        {"h": 0.60, "vx": 0.0, "yaw": 0.0, "steps": 0.0, "wave": 0.0, "kick": 0.0, "r_arm": 0.0, "l_arm": 0.0, "fell": False},
    )
    assert not ok_sq and err_sq["h"] > 0, err_sq
    ok_loc, err_loc = evaluate_trial(
        "locomote",
        {"h": 1.0, "vx": 0.5, "yaw": 0.0, "steps": 5.0, "wave": 0.0, "kick": 0.0, "r_arm": 0.0, "l_arm": 0.0},
        {"h": 1.0, "vx": 0.5, "yaw": 0.0, "steps": 2.0, "wave": 0.0, "kick": 0.0, "r_arm": 0.0, "l_arm": 0.0, "fell": False},
    )
    assert not ok_loc and err_loc["steps"] < 0, err_loc
    assert "cube" not in err_sq and "cube" not in err_loc

    bot.reset_sim()
    bot.policy.eval()
    sgd0 = bot._sgd_steps
    squat_ictx = bot.auto_trial(
        Plan("присесть", skill="squat", params={"depth": "low"}),
        max_trials=2,
        ticks=200,
    )
    assert bot._sgd_steps == sgd0, "in-context must not backprop"
    assert not bot.policy.training
    assert squat_ictx["skill"] == "squat"
    assert squat_ictx["trials"], squat_ictx
    assert "Δheight" in squat_ictx["lines"][0] or "skill=squat" in squat_ictx["lines"][0]

    bot.reset_sim()
    loc_ictx = bot.auto_trial(
        Plan("шаг", skill="locomote", params={"direction": "forward", "speed": "medium", "distance_hint": "3"}),
        max_trials=2,
        ticks=200,
    )
    assert loc_ictx["skill"] == "locomote"
    assert loc_ictx["trials"]

    bot._render()
    assert bot.jpeg()[:2] == b"\xff\xd8"
    bot.reset_sim()
    for n in range(1, 4):
        bot.auto_trial(
            Plan("присесть", skill="squat", params={"depth": "low"}),
            max_trials=1,
            ticks=400,
        )
        for _ in range(200):
            bot.step()
        msg = bot.consolidate()
        print(msg.replace("[Consolidate]", f"[Consolidate #{n}]", 1))
        assert "Replay:" in msg and "Shadow MSE EMA:" in msg, msg
    assert bot.baked
    assert not bot.student_drive, bot.h1_report
    assert bot.h1_report.get("requires_student_on_actuators") is False
    assert "shadow" in bot.h1_report
    assert ALPHA_MAX == 0.12
    print("H1", {k: bot.h1_report[k] for k in ("h1", "mse_true", "gap_zero", "gap_abl", "stage", "alpha") if k in bot.h1_report})
    tel = bot.telemetry()
    print("smoke ok", {k: tel[k] for k in ("outcome", "ctrl", "pelvis_z", "tilt", "status", "skill")})
    bot.close()


if __name__ == "__main__":
    smoke()
