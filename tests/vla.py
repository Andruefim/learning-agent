"""System 1.5 retarget, the world-moved label, and System 1's image/proprio path."""

from __future__ import annotations

import numpy as np
import torch

from agent.config import ERROR_LEN, INSTR_BYTES, STAND_Z, Z_DIM
from agent.h2 import ACTION_DIM, N_ACT
from agent.l3_cmd import CMD_H, CMD_VX, CMD_VY, CMD_WZ
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


if __name__ == "__main__":
    test_retarget_stays_inside_the_command()
    test_a_hinge_swing_counts_as_motion()
    test_image_and_proprio_reach_the_loss()
    test_chunk_plays_without_the_weights()
    print("ok")
