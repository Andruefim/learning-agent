"""Teacher tracker: stand, gait, hip-yaw turn, brace. Control-rate, not L1."""

from __future__ import annotations

import numpy as np
import mujoco

from agent.config import (
    BRACE_TILT,
    BRACE_TILT_WIN,
    FALL_Z,
    HX_PD_CLIP,
    HX_PD_CLIP_UNLOAD,
    PARAM_OK,
    PIVOT_HX_OPEN,
    PIVOT_HX_SHIFT,
    STAND_Z,
    TRACK_OK,
)
from agent.h2 import HIP_YAW_LIM, L_AK, L_EL, L_HX, L_HY, L_HZ, L_KN, L_SH, L_SX, R_AK, R_EL, R_HX, R_HY, R_HZ, R_KN, R_SH, R_SX, ARM_RAISE, STAND_Q, SQUAT_Q, WAIST_P
from agent.plan import Plan, wrap_angle


class TrackerMixin:
    def _pose_from_plan(self, plan: Plan) -> np.ndarray:
        t = plan.teacher()
        height = float(np.clip(t.height + float(self._exec_bias.get("h", 0.0)), 0.0, 1.0))
        q = (1.0 - height) * SQUAT_Q + height * STAND_Q
        wave = float(np.clip(t.wave, 0.0, 1.0))
        r_arm = float(np.clip(t.r_arm, 0.0, 1.0))
        l_arm = float(np.clip(t.l_arm, 0.0, 1.0))
        if wave > 0.15 and r_arm < 0.15 and l_arm < 0.15:
            r_arm = 0.85
        q[R_SH] = float(np.clip(ARM_RAISE * r_arm, self.lo[R_SH], self.hi[R_SH]))
        q[R_EL] = float(np.clip(0.90 * r_arm, self.lo[R_EL], self.hi[R_EL]))
        q[L_SH] = float(np.clip(ARM_RAISE * l_arm, self.lo[L_SH], self.hi[L_SH]))
        q[L_EL] = float(np.clip(0.90 * l_arm, self.lo[L_EL], self.hi[L_EL]))
        q[R_SX] = float(np.clip(-1.20 * t.r_out, self.lo[R_SX], self.hi[R_SX]))
        q[L_SX] = float(np.clip(1.20 * t.l_out, self.lo[L_SX], self.hi[L_SX]))
        if wave > 0.12:
            self._wave_phase += 0.10 * wave
            osc = 0.70 * wave * float(np.sin(self._wave_phase)) * self._stability_scale()
            if r_arm > 0.2:
                q[R_SH] = float(np.clip(q[R_SH] + osc, self.lo[R_SH], self.hi[R_SH]))
                q[R_EL] = float(np.clip(q[R_EL] + 0.25 * osc, self.lo[R_EL], self.hi[R_EL]))
            if l_arm > 0.2:
                q[L_SH] = float(np.clip(q[L_SH] + osc, self.lo[L_SH], self.hi[L_SH]))
                q[L_EL] = float(np.clip(q[L_EL] + 0.25 * osc, self.lo[L_EL], self.hi[L_EL]))
        vx = self._active_vx() if plan is self.waypoint else float(t.vx)
        yaw_cmd = float(np.clip(float(t.yaw) + float(self._exec_bias.get("yaw", 0.0)), -1.0, 1.0))
        if plan is self.waypoint:
            yaw_cmd = self._yaw_from_progress(yaw_cmd)
            self._yaw_applied = yaw_cmd
            yaw = yaw_cmd
        else:
            yaw = yaw_cmd
        walking = abs(vx) > 0.08
        if walking:
            self._phase += 0.09 * float(np.clip(abs(vx), 0.0, 1.0))
            s = float(np.sin(self._phase))
            if self._prev_swing <= 0.0 < s:
                self._step_count += 1
                self._steps_done += 1
            self._prev_swing = s
            swing = 0.22 * abs(vx)
            sign_v = float(np.sign(vx)) if abs(vx) > 1e-6 else 1.0
            if s >= 0:
                q[R_HY] -= swing * sign_v * abs(s)
                q[R_KN] += 0.35 * swing * abs(s)
                q[R_AK] += 0.07 * vx
            else:
                q[L_HY] -= swing * sign_v * abs(s)
                q[L_KN] += 0.35 * swing * abs(s)
                q[L_AK] += 0.07 * vx
        q = self._apply_turn(q, yaw, walking=walking)
        kick = float(np.clip(t.kick, -1.0, 1.0))
        if abs(kick) > 0.2:
            if self._kick_phase < np.pi:
                self._kick_phase += 0.12 * abs(kick)
            a = abs(float(np.sin(min(self._kick_phase, np.pi))))
            if kick > 0:
                q[R_HY] -= 0.32 * a
                q[R_KN] += 0.38 * a
                q[L_HY] += 0.12 * a
                q[L_AK] += 0.10 * a
                q[R_AK] -= 0.06 * a
            else:
                q[L_HY] -= 0.32 * a
                q[L_KN] += 0.38 * a
                q[R_HY] += 0.12 * a
                q[R_AK] += 0.10 * a
                q[L_AK] -= 0.06 * a
        else:
            self._kick_phase = 0.0
        return np.clip(q.astype(np.float32), self.lo, self.hi)

    def _achieved_yaw(self) -> float:
        if self._turn_heading0 is None:
            return 0.0
        return wrap_angle(self._heading() - float(self._turn_heading0))

    def _turn_done(self) -> bool:
        req = self._requested_yaw
        if req is None:
            return False
        return abs(float(req) - self._achieved_yaw()) <= PARAM_OK["yaw"]

    def _stability_scale(self) -> float:
        tilt = self._tilt_up()
        band = max(1e-3, 1.0 - BRACE_TILT)
        tilt_scale = float(np.clip((tilt - BRACE_TILT) / band, 0.0, 1.0))
        e = float(np.linalg.norm(self._balance_err()[:2]))
        err_scale = float(np.clip(1.0 - e / max(4.0 * TRACK_OK, 1e-3), 0.0, 1.0))
        return float(min(tilt_scale, err_scale))

    def _yaw_from_progress(self, yaw_cmd: float) -> float:
        scale = self._stability_scale()
        yaw_cmd = float(np.clip(yaw_cmd * scale, -1.0, 1.0))
        req = self._requested_yaw
        if req is None:
            return yaw_cmd
        rem = float(req) - self._achieved_yaw()
        if abs(rem) <= PARAM_OK["yaw"]:
            return 0.0
        intensity = max(abs(yaw_cmd), scale)
        return float(np.sign(rem)) * float(np.clip(intensity, 0.0, 1.0))

    def _hx_lean_lim(self) -> float:
        return float(min(self.hi[R_HX], -self.lo[R_HX], self.hi[L_HX], -self.lo[L_HX], 0.45))

    def _apply_hip_twist(self, q: np.ndarray, yaw: float) -> np.ndarray:
        hz = float(HIP_YAW_LIM)
        q[R_HZ] = float(np.clip(-hz * yaw, self.lo[R_HZ], self.hi[R_HZ]))
        q[L_HZ] = float(np.clip(-hz * yaw, self.lo[L_HZ], self.hi[L_HZ]))
        hx = 0.10 * self._hx_lean_lim() * yaw
        q[R_HX] -= hx
        q[L_HX] += hx
        return q

    def _apply_pivot_step(self, q: np.ndarray, yaw: float) -> np.ndarray:
        dt = float(self.model.opt.timestep)
        period = float(getattr(self, "_pivot_period", 1.0))
        self._phase += 2.0 * np.pi * dt / max(period, 0.25)
        s = float(np.sin(self._phase))
        self._prev_swing = s
        hx_lim = self._hx_lean_lim()
        frac = float(getattr(self, "_pivot_hx_frac", PIVOT_HX_SHIFT))
        if self._requested_yaw is None:
            frac = min(frac, float(PIVOT_HX_OPEN))
        lean = frac * hx_lim * s * self._stability_scale()
        q[R_HX] += lean
        q[L_HX] += lean
        # Hip yaw turns the whole foot with the leg. Stance stays near 0; swing yaws.
        q[R_HZ] = float(np.clip(-HIP_YAW_LIM * yaw * max(s, 0.0), self.lo[R_HZ], self.hi[R_HZ]))
        q[L_HZ] = float(np.clip(-HIP_YAW_LIM * yaw * max(-s, 0.0), self.lo[L_HZ], self.hi[L_HZ]))
        return q

    def _apply_turn(self, q: np.ndarray, yaw: float, *, walking: bool) -> np.ndarray:
        if abs(yaw) <= 0.08:
            self.turn_mode = "none"
            self._restore_feet()
            return q
        remaining = None
        if self._requested_yaw is not None:
            remaining = abs(float(self._requested_yaw) - self._achieved_yaw())
        twist_budget = float(HIP_YAW_LIM)
        need_pivot = (
            (not walking)
            and (not self._force_twist_only)
            and (remaining is None or remaining > twist_budget)
        )
        if need_pivot:
            self.turn_mode = "pivot"
            self.turn_mechanism = "pivot-step"
            return self._apply_pivot_step(q, yaw)
        self.turn_mode = "twist"
        self._restore_feet()
        if self.turn_mechanism == "unknown":
            self.turn_mechanism = "foot-slip-twist"
        return self._apply_hip_twist(q, yaw)

    def _should_brace(self, err: np.ndarray) -> tuple[bool, str]:
        tilt = self._tilt_up()
        pelvis_z = float(self._pelvis()[2])
        e_xy = float(np.linalg.norm(err[:2]))
        self._err_xy_hist.append(e_xy)
        self._tilt_hist.append(tilt)
        if tilt < BRACE_TILT:
            return True, "brace"
        if pelvis_z < self._floor_z():
            return True, "failed"
        if e_xy >= 6.0 * TRACK_OK and tilt < 0.80:
            return True, "brace"
        if len(self._tilt_hist) >= BRACE_TILT_WIN:
            tilt_drop = float(self._tilt_hist[0]) - tilt
            if tilt < 0.80 and tilt_drop >= 0.25 * (1.0 - BRACE_TILT):
                return True, "brace"
        if len(self._err_xy_hist) >= 20:
            med = float(np.median(np.asarray(self._err_xy_hist, dtype=np.float32)))
            e_old = float(self._err_xy_hist[-20])
            if tilt < 0.80 and e_xy > med + 2.0 * TRACK_OK and (e_xy - e_old) > TRACK_OK:
                return True, "brace"
        return False, ""

    def _sagittal_pd(self, err_x: float, d_x: float) -> tuple[float, float, float]:
        """Ankle + hip strategy. H2 box feet: COM forward → +ankle_pitch, +hip_pitch.

        Toy-humanoid sag_ak used −1.2·err (plantarflex by tilting the sole); on H2
        that dumps the foot box off its heel. Empirically +P −D holds a 70 kg,
        ~1 m COM inverted pendulum. Hip uses the same sign pattern with larger
        gain (τ_hip max ~360 Nm vs ankle ~67 Nm). Brace uses higher gain and a
        wider clip than track.
        """
        brace = self.outcome == "brace"
        ak_lim = float(np.clip(0.14 + 0.8 * abs(err_x), 0.14, 0.28))
        hy_lim = float(np.clip(0.20 + 1.5 * abs(err_x), 0.20, 0.50))
        if brace:
            ak_lim = min(ak_lim + 0.04, 0.32)
            hy_lim = min(hy_lim + 0.10, 0.58)
        if brace:
            sag_ak = float(np.clip(3.2 * err_x - 6.5 * d_x, -ak_lim, ak_lim))
            sag_hy = float(np.clip(5.5 * err_x - 10.0 * d_x, -hy_lim, hy_lim))
            z_err = STAND_Z - float(self._pelvis()[2])
            sag_hy = float(np.clip(sag_hy + 1.2 * max(z_err, 0.0), -hy_lim, hy_lim))
        else:
            sag_ak = float(np.clip(2.4 * err_x - 5.0 * d_x, -ak_lim, ak_lim))
            sag_hy = float(np.clip(4.0 * err_x - 8.0 * d_x, -hy_lim, hy_lim))
            h = float(self.waypoint.teacher().height)
            scale = float(np.clip(h, 0.5, 1.0))
            sag_ak *= scale
            sag_hy *= scale
        # Waist stays out of the sagittal loop: waist_pitch ROM is ±0.44 and
        # collinear with hip_pitch, so it double-counts torso rotation.
        sag_wp = 0.0
        return sag_ak, sag_hy, sag_wp

    def _maybe_unbrace(self, err: np.ndarray, tilt: float, pelvis_z: float, e: float) -> None:
        if self.outcome != "brace" or self.status == "failed":
            return
        if tilt > 0.90 and pelvis_z >= STAND_Z - 0.03 and e < 5.5 * TRACK_OK:
            self.outcome = "ok"
            self.status = "ok"

    def _track(self, err: np.ndarray) -> tuple[np.ndarray, str]:
        tilt = self._tilt_up()
        pelvis_z = float(self._pelvis()[2])
        e = float(np.linalg.norm(err[:2]))
        if tilt > BRACE_TILT and pelvis_z > FALL_Z and e < TRACK_OK:
            self._was_tracking = True
        if self.outcome != "brace":
            hit, why = self._should_brace(err)
            if hit:
                self.outcome = "brace"
                self.status = why
                self.user_cmd = ""
        self._tilt_prev = tilt
        plan = Plan.stand() if self.outcome == "brace" else self.waypoint
        q = self._pose_from_plan(plan)
        if (
            self.outcome != "brace"
            and self._steps_goal > 0
            and self._step_count >= self._steps_goal
        ):
            self._steps_goal = 0
            self._step_count = 0
            if self.user_cmd:
                self.user_cmd = ""
                self.intent = ""
                self.status = "done"
        d_off = err[:2] - self._off_prev
        self._off_prev = err[:2].copy()
        yaw = float(self._heading())
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        err_f = c * float(err[0]) + s * float(err[1])
        d_f = c * float(d_off[0]) + s * float(d_off[1])
        sag_ak, sag_hy, sag_wp = self._sagittal_pd(err_f, d_f)
        if self.outcome != "brace":
            vx = abs(self._active_vx())
            if vx > 0.08:
                fade = float(np.clip(1.0 - (vx - 0.08) / 0.42, 0.2, 1.0))
                sag_ak *= fade
                sag_hy *= fade
                sag_wp *= fade
        q[R_AK] += sag_ak
        q[L_AK] += sag_ak
        q[R_HY] += sag_hy
        q[L_HY] += sag_hy
        q[WAIST_P] += sag_wp
        self._maybe_unbrace(err, tilt, pelvis_z, e)
        pd_hx = 1.2 * err[1] - 6.0 * d_off[1]
        q[R_HX] += float(np.clip(pd_hx, *self._hx_pd_room(q, R_HX)))
        q[L_HX] += float(np.clip(pd_hx, *self._hx_pd_room(q, L_HX)))
        src = "brace" if self.outcome == "brace" else "track"
        return np.clip(q, self.lo, self.hi), src

    def _hx_pd_room(self, q: np.ndarray, idx: int) -> tuple[float, float]:
        """PD hip-x clip = leftover ROM after the pose, never more than HX_PD_CLIP."""
        lo_room = float(self.lo[idx] - q[idx])
        hi_room = float(self.hi[idx] - q[idx])
        cap = float(HX_PD_CLIP_UNLOAD if self.turn_mode == "pivot" else HX_PD_CLIP)
        return (max(lo_room, -cap), min(hi_room, cap))

    def _contact_normal(self, geom_ids) -> float:
        if isinstance(geom_ids, (int, np.integer)):
            want = {int(geom_ids)}
        else:
            want = {int(g) for g in geom_ids}
        total = 0.0
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            if int(con.geom1) not in want and int(con.geom2) not in want:
                continue
            frc = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, frc)
            total += abs(float(frc[0]))
        return total

    def probe_pivot_unload(self, shift_frac: float, *, ticks: int = 700) -> dict:
        """Measure swing-foot F_N via mj_contactForce and hip-x room vs PD."""
        prev_frac = float(self._pivot_hx_frac)
        try:
            self.reset_sim()
            for _ in range(180):
                self.step()
            self._pivot_hx_frac = float(shift_frac)
            self.apply_plan(
                Plan("поверни налево на 90", skill="turn", params={"direction": "left", "angle": "90"}),
                fresh=True,
                l1_ok=True,
            )
            rec: list[dict] = []
            n_brace = 0
            for _ in range(int(ticks)):
                self.step()
                if self.outcome == "brace":
                    n_brace += 1
                if self.turn_mode != "pivot" or self._tilt_up() < 0.7:
                    continue
                s = float(np.sin(self._phase))
                swing_g = self.r_foot_geoms if s >= 0.0 else self.l_foot_geoms
                stance_g = self.l_foot_geoms if s >= 0.0 else self.r_foot_geoms
                qhx = float(self._last_teacher[R_HX])
                rec.append(
                    {
                        "swing": self._contact_normal(swing_g),
                        "stance": self._contact_normal(stance_g),
                        "q_rhx": qhx,
                        "room_hi": float(self.hi[R_HX]) - qhx,
                        "room_lo": qhx - float(self.lo[R_HX]),
                        "phase": abs(s),
                        "tilt": self._tilt_up(),
                        "err": float(np.linalg.norm(self._balance_err()[:2])),
                    }
                )
            if not rec:
                return {"n": 0, "shift_frac": float(shift_frac), "n_brace": n_brace}
            sw = np.asarray([r["swing"] for r in rec], dtype=np.float64)
            st = np.asarray([r["stance"] for r in rec], dtype=np.float64)
            q = np.asarray([r["q_rhx"] for r in rec], dtype=np.float64)
            room = np.asarray([min(r["room_hi"], r["room_lo"]) for r in rec], dtype=np.float64)
            peak = [r for r in rec if r["phase"] >= 0.7]
            psw = np.asarray([r["swing"] for r in peak], dtype=np.float64) if peak else sw
            pst = np.asarray([r["stance"] for r in peak], dtype=np.float64) if peak else st
            return {
                "n": len(rec),
                "n_peak": len(peak),
                "shift_frac": float(shift_frac),
                "swing_fn_med": float(np.median(sw)),
                "swing_fn_mean": float(np.mean(sw)),
                "stance_fn_med": float(np.median(st)),
                "ratio": float(np.median(sw) / max(float(np.median(st)), 1e-6)),
                "peak_swing_fn_med": float(np.median(psw)),
                "peak_stance_fn_med": float(np.median(pst)),
                "peak_ratio": float(np.median(psw) / max(float(np.median(pst)), 1e-6)),
                "q_rhx_min": float(np.min(q)),
                "q_rhx_max": float(np.max(q)),
                "hi": float(self.hi[R_HX]),
                "lo": float(self.lo[R_HX]),
                "min_room": float(np.min(room)),
                "med_room": float(np.median(room)),
                "sat_frac": float(np.mean(room < 0.02)),
                "n_brace": int(n_brace),
                "tilt_med": float(np.median([r["tilt"] for r in rec])),
                "err_med": float(np.median([r["err"] for r in rec])),
            }
        finally:
            self._pivot_hx_frac = prev_frac
            self._restore_feet()

