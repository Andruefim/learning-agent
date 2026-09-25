"""Where the head ray lands at spawn, and whether that stance walks into the room."""

import time

import mujoco
import numpy as np

from agent.h2 import MODEL_XML, SPAWN_Z, STAND_Q
from agent.reach import (
    LOOK_H,
    LOOK_W,
    camera_basis,
    ray_direction,
    ray_hit,
    solve_arm,
)


def stand():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = SPAWN_Z
    data.qpos[3] = 1.0
    for i in range(len(STAND_Q)):
        jid = int(model.actuator_trnid[i, 0])
        data.qpos[int(model.jnt_qposadr[jid])] = STAND_Q[i]
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = stand()
    for name in (
        "toaster_main_group_main",
        "knife_block_main_group_main",
        "pelvis",
    ):
        bid = int(model.body(name).id)
        print(name, np.round(data.xpos[bid], 3))
    origin, rot, fovy = camera_basis(model, data)
    pelvis = int(model.body("pelvis").id)
    aspect = LOOK_W / LOOK_H
    qadr = np.array([int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]) for i in range(29)])
    vadr = np.array([int(model.jnt_dofadr[int(model.actuator_trnid[i, 0])]) for i in range(29)])
    for u, v in ((0.5, 0.5), (0.92, 0.65), (0.9, 0.6)):
        direction = ray_direction(rot, fovy, aspect, u, v)
        hit = ray_hit(model, data, origin, direction, pelvis)
        if hit is None:
            print(f"uv {u} {v} miss")
            continue
        point, body = hit
        print(
            f"uv {u:.2f} {v:.2f} body {model.body(body).name} point {np.round(point, 3)} dist {np.linalg.norm(point - origin):.2f}"
        )
        goal = {"hand": "right", "u": u, "v": v, "point": point, "body": body}
        t0 = time.perf_counter()
        sol, gap = solve_arm(model, data, goal, qadr, vadr)
        dt = time.perf_counter() - t0
        hang = np.asarray(data.qpos[qadr[22:29]])
        arm = np.asarray(sol[22:29])
        print(
            f"  {dt:.2f}s gap {gap:.3f} shift {None if goal.get('shift') is None else np.round(goal['shift'], 3)} arm {np.round(arm - hang, 2)}"
        )
    renderer = mujoco.Renderer(model, 240, 320)
    renderer.update_scene(data, camera="head")
    from PIL import Image
    rgb = renderer.render()
    Image.fromarray(rgb).save(r"C:\Users\Andrey\Desktop\agi\learning-agent\tests\_head.jpg")
    renderer.close()
    print("head saved")


if __name__ == "__main__":
    main()
