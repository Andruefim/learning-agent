import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("models/unitree_h2/scene_app.xml")
d = mujoco.MjData(m)
print("nq", m.nq, "nu", m.nu, "nv", m.nv, "dt", m.opt.timestep)
for i in range(m.nu):
    jid = int(m.actuator_trnid[i, 0])
    print(i, m.actuator(i).name, "range", m.jnt_range[jid].tolist())
print("cameras", [m.camera(i).name for i in range(m.ncam)])
stand = np.zeros(m.nu, np.float64)
for i in (0, 6):
    stand[i] = -0.4
for i in (3, 9):
    stand[i] = 0.8
for i in (5, 11):
    stand[i] = -0.4
d.qpos[:] = 0
d.qpos[2] = 1.10
d.qpos[3] = 1
d.qpos[7:] = stand
mujoco.mj_forward(m, d)
print("pelvis z", float(d.xpos[m.body("pelvis").id, 2]))
print("torso z", float(d.xpos[m.body("torso_link").id, 2]))
lp = m.body("left_ankle_pitch_link").id
print("L ankle body z", float(d.xpos[lp, 2]))
adr, n = int(m.body_geomadr[lp]), int(m.body_geomnum[lp])
for g in range(adr, adr + n):
    print(" geom", g, "type", int(m.geom_type[g]), "z", float(d.geom_xpos[g, 2]), "size", m.geom_size[g, :3].tolist())
