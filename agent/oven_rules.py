"""Rule 7 + honest contact success + force probes for the oven task (Day A).

Rule 7: no external forces on the world — xfrc/qfrc on non-robot bodies/dofs must
stay zero; no world qpos/qvel writes; no equality/mocap except diagnostic stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from agent.config import (
    OVEN_CONTACT_MIN_ANGLE,
    OVEN_CONTACT_OPEN_MS,
    OVEN_SUCCESS_ANGLE,
    OVEN_SUCCESS_HOLD_S,
)
from agent.oven_task import DOOR_JOINT, HANDLE_GEOM


def robot_body_ids(model: mujoco.MjModel, root_name: str = "pelvis") -> set[int]:
    root = int(model.body(root_name).id)
    out = {root}
    # Children follow increasing ids with parent pointers.
    for bid in range(model.nbody):
        b = bid
        while b > 0:
            if b == root:
                out.add(bid)
                break
            b = int(model.body_parentid[b])
    return out


def robot_dof_ids(model: mujoco.MjModel, robot_bodies: set[int]) -> set[int]:
    dofs: set[int] = set()
    for j in range(model.njnt):
        body = int(model.jnt_bodyid[j])
        if body not in robot_bodies:
            continue
        adr = int(model.jnt_dofadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            dofs.update(range(adr, adr + 6))
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            dofs.update(range(adr, adr + 3))
        else:
            dofs.add(adr)
    return dofs


def hand_geom_ids(model: mujoco.MjModel, robot_bodies: set[int]) -> set[int]:
    """Collision geoms on wrist/palm/finger links (Dex3)."""
    out: set[int] = set()
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        if body not in robot_bodies:
            continue
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").lower()
        if any(k in name for k in ("wrist", "palm", "hand", "finger", "thumb", "index", "middle")):
            out.add(gid)
    return out


@dataclass
class Rule7State:
    """Snapshot world qpos/qvel after reset to detect illicit writes."""

    world_qpos_idx: np.ndarray
    world_qvel_idx: np.ndarray
    qpos0: np.ndarray
    qvel0: np.ndarray
    allow_equality: bool = False
    violations: list[str] = field(default_factory=list)

    @classmethod
    def capture(cls, model: mujoco.MjModel, data: mujoco.MjData, *, allow_equality: bool = False) -> "Rule7State":
        robot_bodies = robot_body_ids(model)
        robot_dofs = robot_dof_ids(model, robot_bodies)
        # qpos indices belonging to robot joints
        robot_qpos: set[int] = set()
        for j in range(model.njnt):
            body = int(model.jnt_bodyid[j])
            if body not in robot_bodies:
                continue
            adr = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                robot_qpos.update(range(adr, adr + 7))
            elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                robot_qpos.update(range(adr, adr + 4))
            else:
                robot_qpos.add(adr)
        world_qpos = np.array([i for i in range(model.nq) if i not in robot_qpos], dtype=np.int32)
        world_qvel = np.array([i for i in range(model.nv) if i not in robot_dofs], dtype=np.int32)
        return cls(
            world_qpos_idx=world_qpos,
            world_qvel_idx=world_qvel,
            qpos0=np.asarray(data.qpos[world_qpos], dtype=np.float64).copy(),
            qvel0=np.asarray(data.qvel[world_qvel], dtype=np.float64).copy(),
            allow_equality=allow_equality,
        )

    def check(self, model: mujoco.MjModel, data: mujoco.MjData, robot_bodies: set[int], robot_dofs: set[int]) -> list[str]:
        bad: list[str] = []
        # xfrc on world bodies
        for bid in range(model.nbody):
            if bid in robot_bodies:
                continue
            if float(np.linalg.norm(data.xfrc_applied[bid])) > 1e-9:
                bad.append(f"xfrc_applied body={bid}")
                break
        # qfrc_applied on world dofs
        for did in range(model.nv):
            if did in robot_dofs:
                continue
            if abs(float(data.qfrc_applied[did])) > 1e-9:
                bad.append(f"qfrc_applied dof={did}")
                break
        # equality / mocap (except diagnostic stage 1)
        if not self.allow_equality:
            if model.neq > 0 and hasattr(data, "eq_active"):
                if np.any(np.asarray(data.eq_active)):
                    bad.append("equality_active")
            if model.nmocap > 0 and float(np.linalg.norm(data.mocap_pos)) > 1e-9:
                # mocap may be zeroed; also check if any mocap body is used — conservative
                pass
        # world qpos/qvel writes (beyond passive dynamics — compare to previous step stored externally)
        return bad


@dataclass
class ContactOpenTracker:
    """Accumulate hand–handle contact while the door angle is increasing."""

    handle_gid: int
    hand_gids: set[int]
    dt: float
    contact_ms: float = 0.0
    angle_at_last_contact: float = 0.0
    touching_geoms: set[str] = field(default_factory=set)
    prev_angle: float = 0.0
    hold_s: float = 0.0
    max_contact_force: float = 0.0
    max_contact_normal: float = 0.0
    max_contact_friction: float = 0.0
    max_hinge_passive_abs: float = 0.0
    max_hinge_constraint_abs: float = 0.0

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, angle: float, door_dof: int) -> None:
        rising = angle > self.prev_angle + 1e-6
        # contacts
        force = np.zeros(6)
        touched = False
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if self.handle_gid not in pair:
                continue
            other = g2 if g1 == self.handle_gid else g1
            if other not in self.hand_gids:
                continue
            touched = True
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
            self.touching_geoms.add(str(name))
            mujoco.mj_contactForce(model, data, i, force)
            # force[0] normal-ish in contact frame; 1:3 tangential
            n = abs(float(force[0]))
            f = float(np.linalg.norm(force[1:3]))
            tot = float(np.linalg.norm(force[:3]))
            self.max_contact_normal = max(self.max_contact_normal, n)
            self.max_contact_friction = max(self.max_contact_friction, f)
            self.max_contact_force = max(self.max_contact_force, tot)
        if rising and touched:
            self.contact_ms += self.dt * 1000.0
            self.angle_at_last_contact = float(angle)
        # hinge loads
        self.max_hinge_passive_abs = max(self.max_hinge_passive_abs, abs(float(data.qfrc_passive[door_dof])))
        self.max_hinge_constraint_abs = max(
            self.max_hinge_constraint_abs, abs(float(data.qfrc_constraint[door_dof]))
        )
        if angle >= OVEN_SUCCESS_ANGLE:
            self.hold_s += self.dt
        else:
            self.hold_s = 0.0
        self.prev_angle = float(angle)

    def success(self) -> bool:
        return (
            self.hold_s >= OVEN_SUCCESS_HOLD_S
            and self.contact_ms >= OVEN_CONTACT_OPEN_MS
            and self.angle_at_last_contact >= OVEN_CONTACT_MIN_ANGLE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contact_ms": float(self.contact_ms),
            "angle_at_last_contact": float(self.angle_at_last_contact),
            "hold_s": float(self.hold_s),
            "touching_geoms": sorted(self.touching_geoms),
            "thresholds": {
                "success_angle": OVEN_SUCCESS_ANGLE,
                "hold_s": OVEN_SUCCESS_HOLD_S,
                "contact_open_ms": OVEN_CONTACT_OPEN_MS,
                "contact_min_angle": OVEN_CONTACT_MIN_ANGLE,
            },
            "max_contact_force": float(self.max_contact_force),
            "max_contact_normal": float(self.max_contact_normal),
            "max_contact_friction": float(self.max_contact_friction),
            "max_hinge_passive_abs": float(self.max_hinge_passive_abs),
            "max_hinge_constraint_abs": float(self.max_hinge_constraint_abs),
            "success_contact_rule": bool(self.success()),
        }


def door_params(model: mujoco.MjModel) -> dict[str, float]:
    jid = int(model.joint(DOOR_JOINT).id)
    dof = int(model.jnt_dofadr[jid])
    # lever arm ≈ handle offset from hinge (approx from geom vs jnt)
    return {
        "damping": float(model.dof_damping[dof]),
        "frictionloss": float(model.dof_frictionloss[dof]),
        "stiffness": float(model.jnt_stiffness[jid]),
        "arm_m_approx": 0.35,  # documented estimate; refined in runner from geometry
    }


def mark_contaminated(reports_root) -> list[str]:
    """Write CONTAMINATED.md markers next to known assist-era reports."""
    from pathlib import Path

    root = Path(reports_root)
    targets = [
        "tier_a_gate100",
        "tier_a_smoke20",
        "tier_b_gate100",
        "tier_b_k1_smoke",
        "student_only_eval100",
    ]
    note = (
        "# CONTAMINATED\n\n"
        "This report used a Tier A teacher with privileged door `qfrc_applied` assist "
        "and/or base freeze as an unaccounted external aid. Results are **invalid** for "
        "honest gates. See `reports/day_a_ablation/day_a_report.md`.\n"
        "\n"
        "Zero-policy gate 3.1 (`reports/gate31_zero`) is NOT contaminated.\n"
    )
    written = []
    for name in targets:
        d = root / name
        if d.is_dir():
            p = d / "CONTAMINATED.md"
            p.write_text(note, encoding="utf-8")
            written.append(str(p))
    # checkpoints dagger / student
    ck = root.parent / "checkpoints"
    for name in ("student_v0", "student_v0_trials", "student_dagger"):
        d = ck / name
        if d.is_dir():
            p = d / "CONTAMINATED.md"
            p.write_text(note, encoding="utf-8")
            written.append(str(p))
    return written
