"""System 1.5 retarget, the world-moved label, and System 1's image/proprio path."""

from __future__ import annotations

import numpy as np
import torch

from agent.config import ERROR_LEN, INSTR_BYTES, STAND_Z, Z_DIM
from agent.h2 import ACTION_DIM, N_ACT
from agent.engine import RobotEngine
from agent.l3_cmd import CMD_H, CMD_VX, CMD_VY, CMD_WZ, stand_command
from agent.s1 import CHUNK, CommandTransformer
from agent.vla import ARM_DELTA, BASE_VX, HOLD_TICKS, System15, pose_moved, retarget_action


def _limits():
    return np.full(14, -2.0, np.float32), np.full(14, 2.0, np.float32)


def test_retarget_stays_inside_the_command():
    action = np.zeros(32, np.float32)
    action[0] = 5.0
    action[14] = 1.0
    action[15] = 1.0
    action[16] = 0.05
    lo, hi = _limits()
    cmd = retarget_action(action, np.zeros(14, np.float32), lo, hi)
    assert cmd.shape == (ACTION_DIM,)
    assert abs(float(cmd[4]) - ARM_DELTA) < 1e-5
    assert abs(float(cmd[CMD_VX]) - BASE_VX) < 1e-5
    assert float(cmd[CMD_VY]) == 0.0
    assert float(cmd[CMD_WZ]) == 0.0
    assert abs(float(cmd[CMD_H]) - float(STAND_Z)) < 1e-5
    assert np.all(cmd[4:18] <= hi + 1e-5) and np.all(cmd[4:18] >= lo - 1e-5)


def test_a_hinge_swing_counts_as_motion():
    x = np.zeros((1, 3), np.float64)
    eye = np.zeros((1, 9), np.float64)
    eye[0, 0] = eye[0, 4] = eye[0, 8] = 1.0
    assert pose_moved(x, eye, x, eye, []) is False
    assert pose_moved(x, eye, x, eye, [0]) is False
    shifted = x.copy()
    shifted[0, 0] = 0.03
    assert pose_moved(x, eye, shifted, eye, [0]) is True
    swung = eye.copy()
    th = 0.1
    c, s = float(np.cos(th)), float(np.sin(th))
    swung[0] = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    assert pose_moved(x, eye, x, swung, [0]) is True


def test_image_and_proprio_reach_the_loss():
    net = CommandTransformer()
    b = 2
    image = torch.rand(b, 3, 48, 64)
    proprio = torch.rand(b, N_ACT)
    language = torch.zeros(b, INSTR_BYTES, dtype=torch.long)
    z = torch.zeros(b, Z_DIM)
    errors = torch.zeros(b, ERROR_LEN, 3)
    chunk = torch.randn(b, CHUNK, ACTION_DIM)
    with torch.no_grad():
        same = net.sample(image, proprio, language, z, errors, steps=1)
        other = net.sample(torch.rand_like(image), torch.rand_like(proprio), language, z, errors, steps=1)
    assert torch.allclose(same, other)
    loss = net.cfm_loss(image, proprio, language, z, errors, chunk)
    loss.backward()
    assert float(net.see.weight.grad.abs().sum()) > 0.0
    assert float(net.feel.weight.grad.abs().sum()) > 0.0


def test_a_stale_command_drops_speed():
    teacher = System15(torch.device("cpu"))
    held = stand_command()
    held[CMD_VX] = np.float32(0.25)
    held[CMD_WZ] = np.float32(0.40)
    arms = held[4:18].copy()
    teacher._latched = held
    lo, hi = _limits()
    cmd = held
    for _ in range(250):
        cmd = teacher.command(np.zeros(14, np.float32), lo, hi)
    assert abs(float(cmd[CMD_VX])) < 0.08
    assert abs(float(cmd[CMD_WZ])) < 0.12
    assert np.allclose(cmd[4:18], arms)


def test_the_next_look_waits_until_the_arm_stops():
    teacher = System15(torch.device("cpu"))
    teacher._ready = True
    teacher._chunk = np.zeros((10, 32), np.float32)
    teacher._infer_s = 0.2
    teacher._i = 1
    assert teacher.wants_frame() is True
    assert teacher.wants_still_frame() is False
    teacher._chunk = None
    assert teacher.wants_still_frame() is True


def test_the_arm_waits_until_the_body_has_stopped():
    bot = _bare_engine()
    bot.walk = None
    bot.data = None
    bot._approach_arrived = False
    bot._arm_wait0 = None
    bot._queue = [_frame(reach={"u": 0.5, "v": 0.6, "point": np.zeros(3)})]
    bot._queue[0]["approaching"] = True
    assert bot._pi0_task() is True
    assert bot._hands_ready() is False
    bot._approach_arrived = True
    bot._queue[0]["approaching"] = False
    assert bot._hands_ready() is True


def test_arm_waits_without_cast_point():
    bot = _bare_engine()
    bot.walk = None
    bot.data = None
    bot._approach_arrived = True
    bot._queue = [_frame(reach={"u": 0.5, "v": 0.6})]
    bot._queue[0]["approaching"] = False
    assert bot._hands_ready() is False


def test_a_waiting_arm_is_not_a_miss():
    bot = _bare_engine()
    bot._approach_arrived = True
    bot._pi0_driving = False
    bot._grasp_mark = -1
    bot._grasp_tick0 = 0
    bot._tick = 5000
    goal = {"u": 0.5, "v": 0.6, "point": np.zeros(3), "body": 1, "hand": "right"}
    frame = _frame(reach=goal)
    assert bot._grasp_progress(goal, frame) is False
    assert goal.get("stale") is not True
    assert bot._grasp_tick0 is None


def test_verify_repoint_keeps_anchor_near_field():
    bot = _bare_engine()
    bot.lock = __import__("threading").Lock()
    bot._approach_arrived = True
    bot._anchor = {"point": np.array([1.0, 0.0, 1.0]), "body": 3, "u": 0.5, "v": 0.5}
    bot._need_gaze_plan = False
    bot._need_verify = False
    bot.last_l1 = 0.0
    bot._queue = [_frame(reach={"u": 0.5, "v": 0.5, "point": np.zeros(3)})]
    bot._notes = []
    bot._note = lambda m: bot._notes.append(m)
    bot.apply_verify({"done": False, "retry": "repoint"})
    assert bot._need_gaze_plan is False
    assert bot._need_verify is False
    assert any("якорю" in n for n in bot._notes)


def test_wall_cast_restores_anchor():
    bot = _bare_engine()
    bot._gaze_locked = True
    bot.data = type("D", (), {"qpos": np.zeros(50)})()
    bot.qadr = np.arange(29)
    # WAIST_P must be an index into qpos via qadr
    from agent.h2 import WAIST_P

    bot.qadr = np.arange(29)
    bot.data.qpos = np.zeros(40)
    bot._anchor = {
        "point": np.array([0.5, 0.0, 1.0]),
        "aim": np.array([1.0, 0.0, 0.0]),
        "body": 3,
        "u": 0.5,
        "v": 0.5,
        "hand": "right",
        "name": "fridge_door",
    }
    bot._notes = []
    bot._note = lambda m: bot._notes.append(m)
    bot._marker_pos = None
    bot._marker_aim = None
    bot._marker_dim = False
    bot._need_gaze_plan = True
    goal = {"u": 0.5, "v": 0.5}
    # Monkeypatch ray path to hit a wall body
    import agent.engine as eng

    def fake_basis(*a, **k):
        return np.zeros(3), np.eye(3), 45.0

    def fake_dir(*a, **k):
        return np.array([1.0, 0.0, 0.0])

    def fake_hit(*a, **k):
        return np.array([1.0, 0.0, 1.0]), 9

    def fake_root(model, body):
        return int(body)

    bot.model = type("M", (), {})()
    bot.pelvis_id = 0
    bot._body_name = lambda b: "wall_room_main"
    orig = (eng.camera_basis, eng.ray_direction, eng.ray_hit, eng.assembly_root)
    eng.camera_basis, eng.ray_direction, eng.ray_hit, eng.assembly_root = (
        fake_basis,
        fake_dir,
        fake_hit,
        fake_root,
    )
    try:
        bot._cast_look(goal)
    finally:
        eng.camera_basis, eng.ray_direction, eng.ray_hit, eng.assembly_root = orig
    assert goal.get("point") is not None
    assert np.allclose(goal["point"], bot._anchor["point"])
    assert bot._need_gaze_plan is False


def test_fingers_stay_curled_for_raise():
    from agent.dex3 import finger_curl

    bot = _bare_engine()
    bot._queue = [_frame(reach={"u": 0.5, "v": 0.5, "point": np.zeros(3), "hand": "right"})]
    q = bot._finger_command()
    assert float(np.max(np.abs(q - finger_curl("right")))) < 1e-5
    bot._queue = [_frame(reach={})]
    q2 = bot._finger_command()
    assert float(np.max(np.abs(q2 - finger_curl("both")))) < 1e-5


def test_neural_near_after_approach():
    bot = _bare_engine()
    bot._approach_arrived = False
    bot._queue = [_frame(reach={"u": 0.5, "v": 0.6, "point": np.zeros(3)})]
    assert bot._neural_near() is False
    bot._approach_arrived = True
    assert bot._neural_near() is True
    assert bot._pi0_owns_near() is False


def test_the_next_chunk_is_asked_for_early():
    teacher = System15(torch.device("cpu"))
    teacher._ready = True
    teacher._chunk = np.zeros((10, 32), np.float32)
    teacher._infer_s = 0.2
    assert teacher.wants_frame() is False
    teacher._i = 1
    assert teacher.wants_frame() is True


def test_a_queued_chunk_crossfades():
    teacher = System15(torch.device("cpu"))
    lo, hi = _limits()
    old = np.zeros((1, 32), np.float32)
    old[:, 14] = 1.0
    teacher._chunk = old
    teacher._hold = HOLD_TICKS - 1
    nxt = np.zeros((4, 32), np.float32)
    nxt[:, 14] = -1.0
    teacher._queued = nxt
    arm = np.zeros(14, np.float32)
    teacher.command(arm, lo, hi)
    assert teacher._chunk is nxt
    assert teacher._blend_left > 0
    later = teacher.command(arm, lo, hi)
    for _ in range(4):
        later = teacher.command(arm, lo, hi)
    assert float(later[CMD_VX]) > 0.0


def test_the_first_chunk_is_played_whole():
    teacher = System15(torch.device("cpu"))
    lo, hi = _limits()
    nxt = np.zeros((4, 32), np.float32)
    nxt[:, 14] = 1.0
    teacher._queued = nxt
    cmd = teacher.command(np.zeros(14, np.float32), lo, hi)
    assert abs(float(cmd[CMD_VX]) - BASE_VX) < 1e-5


def _bare_engine():
    bot = RobotEngine.__new__(RobotEngine)
    bot._awaiting_plan = False
    bot._ep_armed = True
    bot._ep_slept = False
    bot._ep_failed = False
    bot._trial_busy = False
    bot.outcome = "ok"
    bot._queue_i = 0
    bot._queue_tick0 = 0
    bot._ep_tick0 = 0
    bot._tick = 10
    bot._step_count = 0
    bot._steps_goal = 0
    bot.user_cmd = "возьми"
    bot.intent = "возьми"
    bot.model = type("M", (), {"opt": type("O", (), {"timestep": 0.002})()})()
    return bot


def _frame(**over):
    cmd = stand_command()
    frame = {"cmd": cmd, "hold_s": 4.0, "steps": 0, "yaw": None, "reach": {}}
    frame.update(over)
    return frame


def test_a_finished_command_is_recognized():
    bot = _bare_engine()
    walking = _frame(steps=5)
    walking["cmd"][CMD_VX] = np.float32(0.4)
    bot._queue = [walking]
    bot._steps_goal = 5
    bot._step_count = 5
    assert bot._attempt_done() is True
    assert float(bot._plan_command()[CMD_VX]) == 0.0

    bot = _bare_engine()
    bot._queue = [_frame()]
    bot._queue[0]["cmd"][CMD_VX] = np.float32(0.4)
    bot._tick = 3000
    assert bot._attempt_done() is False

    bot = _bare_engine()
    held = stand_command()
    held[4] = np.float32(0.8)
    bot._cmd = held
    bot._queue = [_frame(reach={"u": 0.2, "v": 0.3, "done": True})]
    bot._queue[0]["cmd"][CMD_VX] = np.float32(0.4)
    bot._tick = 3000
    assert bot._pi0_task() is True
    assert bot._attempt_done() is False

    bot._queue[0]["reach"] = {"u": 0.2, "v": 0.3}
    bot._tick = 10
    assert bot._pi0_task() is True
    assert bot._attempt_done() is False
    bot._awaiting_plan = True
    assert bot._pi0_task() is False
    assert bot._attempt_done() is False
    assert abs(float(bot._plan_command()[CMD_VX]) - 0.4) < 1e-5


def test_chunk_plays_without_the_weights():
    teacher = System15(torch.device("cpu"))
    lo, hi = _limits()
    arm = np.zeros(14, np.float32)
    assert teacher.command(arm, lo, hi) is None
    teacher._chunk = np.zeros((2, 32), np.float32)
    teacher._chunk[:, 0] = 1.0
    played = [teacher.command(arm, lo, hi) for _ in range(HOLD_TICKS * 2)]
    assert all(cmd is not None for cmd in played)
    assert teacher._chunk is None
    held = teacher.command(arm, lo, hi)
    assert held is not None and np.allclose(held, played[-1])
    teacher.reset_chunk()
    assert teacher.command(arm, lo, hi) is None


def test_pi0_device_defaults_to_cuda_when_present(monkeypatch):
    from agent.policy import resolve_pi0_device

    monkeypatch.delenv("PI0_DEVICE", raising=False)
    if torch.cuda.is_available():
        assert resolve_pi0_device().type == "cuda"
    monkeypatch.setenv("PI0_DEVICE", "cpu")
    assert resolve_pi0_device().type == "cpu"
    monkeypatch.setenv("PI0_DEVICE", "cuda")
    if torch.cuda.is_available():
        assert resolve_pi0_device().type == "cuda"
    else:
        assert resolve_pi0_device().type == "cpu"


def test_load_now_refuses_cpu():
    teacher = System15(torch.device("cpu"))
    note = teacher.load_now()
    assert note is not None and "CPU" in note
    assert teacher.error
    assert teacher._ready is False
    assert teacher._thread is None


if __name__ == "__main__":
    test_retarget_stays_inside_the_command()
    test_a_hinge_swing_counts_as_motion()
    test_image_and_proprio_reach_the_loss()
    test_a_stale_command_drops_speed()
    test_the_next_look_waits_until_the_arm_stops()
    test_the_arm_waits_until_the_body_has_stopped()
    test_a_waiting_arm_is_not_a_miss()
    test_the_next_chunk_is_asked_for_early()
    test_a_queued_chunk_crossfades()
    test_the_first_chunk_is_played_whole()
    test_a_finished_command_is_recognized()
    test_chunk_plays_without_the_weights()
    test_load_now_refuses_cpu()
    print("ok")
