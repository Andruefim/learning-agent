"""End-to-end verification of System 0 balance/walk, 6D IK, Gemini->Ollama fallback, SLEEP Wake/Sleep, System 1, and rendering."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform in ("darwin", "win32") else "egl"

from agent import RobotEngine, SleepConfig, SLEEP_ACTION_DIM, SLEEP_HORIZON, SLEEP_IMG_SIZE, SLEEP_PROPRIO_DIM
from agent.system0 import MAT_HORIZ_HOOK, default_raw_action, extract_proprio, raw_to_norm_action, solve_whole_body_ik


def smoke() -> None:
    print("[smoke] Initializing RobotEngine...")
    with tempfile.TemporaryDirectory() as tmpdir, patch("agent.engine.CHECKPOINT_DIR", Path(tmpdir)):
        os.environ["STORAGE_DIR"] = tmpdir
        bot = RobotEngine(headless=False, enable_pi0=False)
        try:
            # 1. Idle standing stability (free pelvis, no teleport)
            for _ in range(40):
                bot.step(10)
            tel = bot.telemetry()
            assert not tel["fell"], f"Robot fell at idle: {tel}"
            assert tel["pelvis_z"] > 0.72, f"Bad stand pelvis_z: {tel}"
            assert tel["tilt"] > 0.88, f"Bad stand tilt: {tel}"
            print(f"[smoke] Stand OK: z={tel['pelvis_z']:.3f}, tilt={tel['tilt']:.3f}, cms={tel['cms_slots_used']}/{tel['cms_slots_total']}")

            # 2. Real RL walking forward via G1LegPolicy
            x0 = float(bot.data.xpos[bot.sys0.pelvis_id, 0])
            walk_cmd = default_raw_action()
            walk_cmd[0] = 0.40
            for _ in range(30):
                bot._step_cog_from_worker(walk_cmd, tuck_left=True, tuck_right=True)
            stand_cmd = default_raw_action()
            for _ in range(12):
                bot._step_cog_from_worker(stand_cmd, tuck_left=True, tuck_right=True)
            x1 = float(bot.data.xpos[bot.sys0.pelvis_id, 0])
            assert x1 > x0 + 0.15, f"Robot did not walk forward cleanly: x0={x0:.3f}, x1={x1:.3f}"
            assert not bot.sys0.fell(), "Robot fell while walking"
            print(f"[smoke] Walk OK: dx={x1 - x0:.3f} m")

            # 3. 6D Whole-Body IK reachability test
            bot.reset_sim()
            handle_target = np.array([bot.data.xpos[bot.sys0.pelvis_id, 0] + 0.42, -0.05, 0.72], dtype=np.float64)
            q_ik, ik_err = solve_whole_body_ik(
                bot.model,
                bot.data,
                bot.sys2.scratch,
                bot.qadr,
                bot.vadr,
                bot.sys0.wrist_r_id,
                handle_target,
                MAT_HORIZ_HOOK,
                squat=0.15,
            )
            assert ik_err < 0.03, f"6D IK error too high: {ik_err:.4f}"
            print(f"[smoke] Whole-Body 6D IK OK: err={ik_err * 100:.2f} cm")

            # 4. Gemini -> Ollama automatic fallback verification
            bot.planner.api_key = "fake-test-key"
            with patch.object(bot.planner, "_call_gemini", side_effect=RuntimeError("503 Service Unavailable")), \
                 patch.object(bot.planner, "_call_ollama", return_value={"mode": "body_skill", "skill": "squat", "params": {"depth": "low"}}) as mock_ol:
                plan = bot.planner.plan_command(np.zeros((64, 64, 3), dtype=np.uint8), "присядь")
                assert mock_ol.called, "Ollama fallback was not triggered on Gemini 503"
                assert plan["mode"] == "body_skill" and plan["skill"] == "squat"
                assert "ollama" in bot.planner.active_backend
            print("[smoke] Gemini 503 -> Ollama fallback OK")

            # 5. SLEEP Confidence Gate + Wake -> NREM Consolidate -> REM Dream -> Save/Load
            head_rgb = bot._get_head_rgb_blocking()
            assert head_rgb.shape == (480, 640, 3)
            pr = extract_proprio(
                bot.data, bot.sys0.pelvis_id, bot.sys0.torso_id, bot.sys0.wrist_r_id, bot.qadr, 0.0
            )
            assert pr.shape == (SLEEP_PROPRIO_DIM,)

            _, sigma_before, _ = bot._infer_sys1(head_rgb, pr, "открой духовку")
            assert sigma_before > bot.tau_think, f"Expected initial sigma > tau_think, got {sigma_before}"

            # Populate wake buffer with realistic smooth chunks toward squat/reach pose
            rgb_128 = np.full((SLEEP_IMG_SIZE, SLEEP_IMG_SIZE, 3), 120, dtype=np.uint8)
            target_raw = default_raw_action()
            target_raw[2] = 0.25
            target_raw[5:12] = q_ik[22:29]
            norm_tgt = raw_to_norm_action(target_raw)
            chunk = np.stack([norm_tgt] * SLEEP_HORIZON, axis=0)
            for _ in range(36):
                bot.wake_buffer.append((rgb_128, pr, "открой духовку", chunk, 1))
            bot.unconsolidated_steps = len(bot.wake_buffer)

            # Enough Wake experience + Deep Sleep consolidation so uncertainty drops below tau_think
            bot._run_wake_update(steps=14, allow_consolidate=False)
            bot.standby_mode = True
            bot._sleep_done_for_batch = False
            bot._pending_skills.add(bot._skill_key("открой духовку"))
            bot.ckpt_path = Path(tmpdir) / "sleep_vla.pt"
            bot._run_deep_sleep()

            assert bot.vla.cms.used_slots() > 1, "Expected CMS slots to expand during sleep"
            assert bot._skill_key("открой духовку") in bot.learner.consolidated_skills
            assert bot._effective_sigma("включи тостер", 0.05) >= 0.85, "Unlearned skill must keep high epistemic uncertainty"
            assert bot.ckpt_path.is_file(), "Checkpoint was not saved after deep sleep"
            assert bot.learner.stats["sleeps"] >= 1, "Consolidation did not run"
            print(
                f"[smoke] SLEEP consolidation+dream OK: slots={bot.vla.cms.used_slots()}/{bot.vla.cms.total_slots()}, "
                f"sleeps={bot.learner.stats['sleeps']}, sigma_before={sigma_before:.3f} -> sigma_after={bot.current_uncertainty:.3f}"
            )
            assert bot.current_uncertainty < sigma_before, "Uncertainty should decrease after Wake/Sleep training"

            # Verify System 1 closed-loop inference & execution stay upright
            chunk_pred, sigma_live, ms_live = bot._infer_sys1(head_rgb, pr, "открой духовку")
            bot.set_standby(False)
            ok_sys1 = bot._run_sys1_closed_loop("открой духовку", chunk_pred)
            assert ok_sys1 and not bot.sys0.fell(), "System 1 execution failed or fell"
            print(f"[smoke] System 1 closed-loop OK: latency={ms_live:.2f} ms, sigma={sigma_live:.3f}, sys1_steps={bot.sys1_count}")

            # Verify checkpoint reload + second-episode Wake update with prev_teacher & anchors active
            slots_before_reload = bot.vla.cms.used_slots()
            bot.vla.cms.blocks[1].reset_slots()
            assert bot.learner.load_checkpoint(bot.ckpt_path)
            assert bot.vla.cms.used_slots() == slots_before_reload
            loss_2nd_ep = bot._run_wake_update(steps=2, allow_consolidate=False)
            assert np.isfinite(loss_2nd_ep), f"Second-episode wake update failed: {loss_2nd_ep}"

            # 6b. Autonomous Fall Recovery (supine + yaw=180 prone floor -> SuperMimic RL -> stable upright stand)
            assert bot.sys0.standup is not None, "Stand-up policy not loaded"
            for rec_name, rpos, rquat in (
                ("supine", [-0.6, 0.0, 0.22], [0.7071, 0.0, -0.7071, 0.0]),
                ("yaw180_prone", [-0.9, 0.0, 0.65], [0.0, 0.7071, 0.0, 0.7071]),
            ):
                bot.reset_sim()
                bot.set_power(False)
                with bot._sim_lock:
                    bot.data.qpos[0:3] = rpos
                    bot.data.qpos[3:7] = rquat
                    bot.data.qvel[:] = 0.0
                    import mujoco
                    mujoco.mj_forward(bot.model, bot.data)
                bot.step(150)
                assert bot.sys0.fell(), f"G1 did not settle into fallen state ({rec_name})"
                bot.set_power(True)
                recovered = bot._ensure_upright(timeout_s=10.0)
                assert recovered, f"Autonomous fall recovery timed out ({rec_name})"
                # Verify post-standup handover stays rock-solid upright for 4s (no wobble / delayed topple)
                for _ in range(200):
                    bot.step(10)
                    assert not bot.sys0.fell() and not bot.sys0.recovering, f"G1 wobbled/fell after {rec_name} standup"
                tel_rec = bot.telemetry()
                assert not tel_rec["fell"] and tel_rec["pelvis_z"] > 0.74 and tel_rec["tilt"] > 0.97, (
                    f"Post-standup stability failed ({rec_name}): {tel_rec}"
                )
            print(f"[smoke] Autonomous fall recovery + 4s post-standup stability OK (z={tel_rec['pelvis_z']:.3f}, tilt={tel_rec['tilt']:.3f})")
            bot.reset_sim()

            # 6c. Failed-episode learning (Episodic Lesson + failure REM sleep, without fake Sys1 consolidation)
            prev_sleeps = bot.total_sleep_cycles
            bad_chunk = np.zeros((SLEEP_HORIZON, SLEEP_ACTION_DIM), dtype=np.float32)
            bad_samples = [(np.zeros((128, 128, 3), dtype=np.uint8), np.zeros(SLEEP_PROPRIO_DIM, dtype=np.float32), "сломанный тест", bad_chunk) for _ in range(4)]
            with (
                patch.object(
                    bot.sys2,
                    "execute",
                    return_value={
                        "ok": False,
                        "reason": "attempts_exhausted",
                        "samples": bad_samples,
                        "failure_info": {
                            "before_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                            "after_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                            "target_pt": np.array([0.8, 0.0, 0.4]),
                            "tried_motions": ["pull_back"],
                            "ik_err_cm": 24.0,
                            "fell": False,
                            "what": "no visible change",
                        },
                    },
                ),
                patch.object(bot.planner, "reflect_failure", return_value="target was too low (IK err 24cm); aim higher at the handle"),
            ):
                bot._execute_command("сломанный тест")
            skey = bot._skill_key("сломанный тест")
            assert len(bot.learner.get_lessons(skey)) >= 1, "Episodic lesson not stored after failure"
            assert len(bot.failure_buffer) >= 4, "Failed trajectory chunks not saved in failure_buffer"
            assert bot.total_sleep_cycles > prev_sleeps, "Post-failure Sleep did not run"
            assert skey not in bot.learner.consolidated_skills, "Failed skill must NOT enter consolidated System 1 skills"
            assert bot._effective_sigma("сломанный тест", 0.10) >= 0.85
            print("[smoke] Post-failure REM sleep & episodic lesson learning OK")

            # 6d. Pretrained ONNX Backflip skill (with automatic fall recovery if needed)
            assert bot.sys0.flip is not None, "Flip policy not loaded"
            bot.reset_sim()
            bot._execute_command("сделай сальто")
            tel_flip = bot.telemetry()
            assert not bot.sys0.flipping and not tel_flip["fell"] and tel_flip["pelvis_z"] > 0.70, (
                f"Backflip post-state failed: {tel_flip}"
            )
            print(f"[smoke] Backflip ONNX skill OK (z={tel_flip['pelvis_z']:.3f}, tilt={tel_flip['tilt']:.3f})")
            bot.reset_sim()

            # 6e. Post-sleep autonomous continuation (retry after failure sleep + standby wake resume + Stop/Reset cancel)
            disp_retry = bot.dispatch_command("автоповтор тест")
            assert disp_retry.get("ok") and bot._active_goal == "автоповтор тест" and bot._goal_retries_left == 2
            with bot._cmd_cv:
                queued_cmd = bot._pending_cmd
                bot._pending_cmd = None
            assert queued_cmd == "автоповтор тест"
            with (
                patch.object(
                    bot.sys2,
                    "execute",
                    return_value={
                        "ok": False,
                        "reason": "attempts_exhausted",
                        "samples": bad_samples,
                        "failure_info": {
                            "before_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                            "after_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                            "target_pt": np.array([0.8, 0.0, 0.4]),
                            "tried_motions": ["pull_back"],
                            "ik_err_cm": 18.0,
                            "fell": False,
                            "what": "no visible change",
                        },
                    },
                ),
                patch.object(bot.planner, "reflect_failure", return_value="move closer before pulling"),
            ):
                bot._execute_command("автоповтор тест")
            assert bot._pending_cmd == "автоповтор тест", "Failed episode did not auto-requeue command after failure sleep"
            assert bot._goal_retries_left == 1

            with bot._cmd_cv:
                bot._pending_cmd = None
            bot.set_standby(True)
            assert bot._active_goal == "автоповтор тест"
            bot.set_standby(False)
            assert bot._pending_cmd == "автоповтор тест", "Exiting standby did not resume active goal"

            bot.dispatch_command("стоп")
            assert bot._active_goal is None and bot._pending_cmd is None
            bot.dispatch_command("второй тест")
            bot.reset_sim()
            assert bot._active_goal is None and bot._pending_cmd is None
            print("[smoke] Post-sleep autonomous continuation & cancel OK")

            # 6f. Already-accomplished visual short-circuit stops redundant retries cleanly
            bot.dispatch_command("открой шкаф")
            with bot._cmd_cv:
                bot._pending_cmd = None
            prev_fail_len = len(bot.failure_buffer)
            prev_wake_len = len(bot.wake_buffer)
            with patch.object(
                bot.planner,
                "plan_command",
                return_value={
                    "mode": "manipulate",
                    "visible": True,
                    "already_done": True,
                    "status": "дверца шкафа уже открыта",
                    "uv": None,
                    "object": "cabinet",
                    "part": "door",
                    "motions": ["pull_back"],
                    "hand": "right",
                    "search_dir": "right",
                    "ok": True,
                },
            ) as mock_plan:
                bot._execute_command("открой шкаф")
            assert mock_plan.called, "6f did not reach System 2 plan_command"
            assert bot._active_goal is None and bot._pending_cmd is None, "Already-done goal was not cleared"
            assert len(bot.failure_buffer) == prev_fail_len, "Already-done treated as failure"
            assert len(bot.wake_buffer) == prev_wake_len, "Already-done wrote empty wake chunk"
            print("[smoke] Already-done visual check & retry stop OK")

            # 6g. Safe System 2 (bumper walk, scratch rehearsal, IK teacher -> wake_buffer), pi0 helpers & 429 cooldown
            import time as _time
            import httpx as _httpx
            from agent.pi0_e2e import build_pi0_state, pi0_chunk_to_g1_norm, pi0_enabled, resolve_pi0_source

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("E2E_ENABLE", None)
                os.environ.pop("PI0_ENABLE", None)
                assert not pi0_enabled(), "Untrained frozen pi0 should default to off until G1 adapter is enabled"
            src_path, is_local = resolve_pi0_source("lerobot/pi0_base")
            assert is_local and Path(src_path).is_dir(), f"Expected cached local pi0_base snapshot, got {src_path}"
            st32 = build_pi0_state(pr, default_raw_action(), 4.0)
            assert st32.shape == (32,)
            mock_pi0_raw = np.zeros((1, 50, 32), dtype=np.float32)
            mock_pi0_raw[0, :, 0] = -0.25
            mock_pi0_raw[0, :, 15] = 0.60
            g1_chunk = pi0_chunk_to_g1_norm(mock_pi0_raw, default_raw_action())
            assert g1_chunk.shape == (SLEEP_HORIZON, SLEEP_ACTION_DIM)
            assert np.all(g1_chunk >= -1.0) and np.all(g1_chunk <= 1.0)

            bot.reset_sim()
            # Test scratch mental rehearsal rejects unreachable extreme point
            safe_bad, _, _ = bot.sys2._rehearse_primitive(
                np.array([4.5, 0.0, 2.2]),
                np.array([1.0, 0.0, 0.0]),
                np.eye(3),
                0.05,
                "push_forward",
            )
            assert not safe_bad, "Scratch rehearsal should reject extreme unreachable target"

            prev_wake = len(bot.wake_buffer)
            mock_plan_ik = {
                "mode": "manipulate",
                "visible": True,
                "already_done": False,
                "status": "",
                "uv": (440.0, 250.0),
                "object": "toaster",
                "part": "lever",
                "motions": ["press_down"],
                "hand": "right",
                "search_dir": "right",
                "ok": True,
            }
            mock_ground_ik = {
                "visible": True,
                "already_done": False,
                "status": "",
                "uv": (440.0, 250.0),
                "object": "toaster",
                "part": "lever",
                "motions": ["press_down"],
                "ok": True,
            }
            with (
                patch.object(bot.planner, "plan_command", return_value=mock_plan_ik),
                patch.object(bot.planner, "ground", return_value=mock_ground_ik),
                patch.object(bot.planner, "judge", return_value={"done": True, "changed": True, "progress": 1.0, "what": "lever pressed", "ok": True}),
            ):
                bot._execute_command("тест тостер")

            assert not bot.sys0.fell(), "Robot fell during bumper-safe walk + IK manipulation"
            assert bot.sys0.walk is None or bot.sys0.walk.holding, "Gait policy failed to hand back to stand hold"
            assert len(bot.wake_buffer) > prev_wake, "Successful System 2 teacher episode was not saved into wake_buffer"
            bot.reset_sim()

            # Verify HTTP 429 triggers alternate Gemini model retry + cooldown -> Ollama fallback
            bot.planner.api_key = "fake-test-key"
            bot.planner._gemini_cooldown_until = 0.0
            req429 = _httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent")
            resp429 = _httpx.Response(429, request=req429, headers={"Retry-After": "0.3"})
            err429 = _httpx.HTTPStatusError("429 Too Many Requests", request=req429, response=resp429)
            with patch("agent.planner.time.sleep", return_value=None), \
                 patch.object(bot.planner, "_call_gemini", side_effect=[err429, err429]) as mock_g429, \
                 patch.object(bot.planner, "_call_ollama", return_value={"mode": "body_skill", "skill": "stand", "params": {}}) as mock_o429:
                out429 = bot.planner.plan_command(np.zeros((64, 64, 3), dtype=np.uint8), "встань")
                assert mock_g429.call_count == 2, f"Expected alt-model retry on 429, got {mock_g429.call_count} calls"
                assert mock_o429.called and out429["skill"] == "stand"
                assert bot.planner._gemini_cooldown_until > _time.monotonic(), "429 cooldown was not armed"
            bot.planner._gemini_cooldown_until = 0.0
            print("[smoke] Safe System 2 (bumper walk, scratch rehearsal, IK -> Wake), pi0 helpers & 429 cooldown OK")

            # 6. HAL switches + rendering MJPEG check
            bot.set_power(False)
            assert bot.telemetry()["state"] == "OFF"
            bot.set_power(True)
            assert bot.telemetry()["state"] == "IDLE"
            bot.orbit_camera(15.0, -5.0, -0.5)
            bot.step(10)
            bot._render_sync()
            assert bot._preview_rgb is not None and float(bot._preview_rgb.std()) > 20.0, "Overview camera occluded"
            assert bot.get_overview_jpeg()[:2] == b"\xff\xd8"
            assert bot.get_head_jpeg()[:2] == b"\xff\xd8"
            print("[smoke] HAL switches & MJPEG render OK")
        finally:
            bot.close()

    # 7. FastAPI WebSocket + HTTP routes check
    from fastapi.testclient import TestClient
    import app as server_app

    with tempfile.TemporaryDirectory() as tmpdir2, \
         patch("agent.engine.CHECKPOINT_DIR", Path(tmpdir2)), \
         patch("agent.engine.pi0_enabled", return_value=False):
        with TestClient(server_app.app) as client:
            r = client.get("/")
            assert r.status_code == 200 and "STAND-BY" in r.text and "POWER ON" in r.text
            h = client.get("/health")
            assert h.status_code == 200 and h.json().get("ok") is True
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "power", "state": False})
                ws.send_json({"type": "power", "state": True})
                ws.send_json({"type": "set_tau", "tau": 0.40})
                got_tel = False
                for _ in range(15):
                    msg = ws.receive_json()
                    if msg.get("type") == "telemetry":
                        got_tel = True
                        if abs(msg.get("tau_think", 0.0) - 0.40) < 1e-6:
                            break
                assert got_tel, "Did not receive telemetry over /ws"
    print("[smoke] FastAPI HTTP + WebSocket UI controls OK")
    print("[smoke] ALL CHECKS PASSED.")


if __name__ == "__main__":
    smoke()
