"""Browser sandbox: standing Unitree H2, L1 intention, gated flow."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.websockets import WebSocketState

from agent import Plan, RobotEngine
from agent.flip_policy import wants_flip
from agent.reach import GRASP_TRIES, attach_reach, merge_grasp_point, plan_wants_grasp_point, strip_planner_pixels

engine: RobotEngine | None = None
_stop = threading.Event()
_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, _thread
    engine = RobotEngine()
    _stop.clear()
    _thread = threading.Thread(target=engine.loop, args=(_stop,), daemon=True)
    _thread.start()
    yield
    _stop.set()
    if _thread:
        _thread.join(timeout=1.5)
    if engine:
        engine.close()


app = FastAPI(title="Humanoid sandbox", lifespan=lifespan)


def _bot() -> RobotEngine:
    assert engine is not None
    return engine


@app.get("/")
def index():
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    t = _bot().telemetry()
    return {"ok": True, **t}


def _mjpeg(get_frame):
    def frames():
        last = None
        while True:
            jpeg = get_frame()
            if jpeg and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            else:
                # A fixed sleep after every frame rounded up to ~15 ms on Windows.
                time.sleep(0.004)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@app.get("/video_feed")
def video_feed():
    return _mjpeg(lambda: _bot().jpeg())


@app.get("/eye_feed")
def eye_feed():
    """Head camera at L1 look size — operator preview, not the 48×64 L2 buffer."""
    return _mjpeg(lambda: _bot().look_jpeg())


async def _resolve_grasp(bot: RobotEngine, cmd: str, plan: Plan, jpeg: bytes) -> None:
    """Vision-only u,v merged into a motion-only plan, then raycast-validated."""
    prior = bot.reach_prior()
    # One vision call per cycle. The free tier 503s when this loops, and the robot
    # stands still until a pixel is accepted.
    attempts = 1
    for _ in range(attempts):
        point_plan, pt_ok = await bot.planner.plan_point(cmd, jpeg, prior=prior)
        if not pt_ok or not merge_grasp_point(plan, point_plan):
            continue
        hg = plan.params.get("hand_goal") if isinstance(plan.params, dict) else None
        if not isinstance(hg, dict):
            continue
        u, v = hg.get("u"), hg.get("v")
        if u is None or v is None:
            continue
        good, body = await asyncio.to_thread(bot.validate_grasp_uv, float(u), float(v))
        if good:
            print(f"grasp cast ok u={float(u):.2f} v={float(v):.2f} body={body}", flush=True)
            return
        print(f"grasp cast reject u={float(u):.2f} v={float(v):.2f} body={body}", flush=True)
        strip_planner_pixels(plan)
        attach_reach(plan, cmd)
        prior = {"u": float(u), "v": float(v), "hand": str(hg.get("hand") or "right")}
    print("grasp: no valid pixel after pointing", flush=True)
    bot._note("Ручка не найдена в кадре. Стоит, повтор через 15 с.")
    bot.defer_l1(15.0)


async def _run_l1(bot: RobotEngine, text: str, *, fresh: bool) -> Plan:
    while not bot.begin_l1():
        await asyncio.sleep(0.05)
    cmd = text.strip()
    try:
        scene = bot.scene_brief()
        jpeg = bot.planner_jpeg()
        if bot.wants_gaze_point():
            plan, ok = await bot.planner.plan_point(cmd, jpeg, prior=bot.reach_prior())
        else:
            plan, ok = await bot.planner.plan(cmd, scene, jpeg)
            if ok and plan_wants_grasp_point(plan):
                await _resolve_grasp(bot, cmd, plan, jpeg)
        if bot.intent == cmd:
            bot.apply_plan(plan, fresh=fresh, l1_ok=ok)
        return plan
    finally:
        bot.end_l1()


async def _run_verify(bot: RobotEngine) -> dict:
    while not bot.begin_l1():
        await asyncio.sleep(0.05)
    cmd = (bot.user_cmd or "").strip()
    try:
        jpeg = bot.planner_jpeg()
        result = await bot.planner.verify(cmd, jpeg)
        if bot.intent == cmd:
            bot.apply_verify(result)
        return result
    finally:
        bot.end_l1()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    bot = _bot()
    try:
        while sock.client_state == WebSocketState.CONNECTED:
            try:
                raw = await asyncio.wait_for(sock.receive_json(), timeout=0.12)
            except asyncio.TimeoutError:
                raw = None
            if raw:
                kind = raw.get("type")
                if kind == "command":
                    text = str(raw.get("text", "")).strip()
                    bot.begin_command(text)
                    if wants_flip(text):
                        await sock.send_json({"type": "log", "text": f"«{text}» → сальто"})
                        continue
                    plan = await _run_l1(bot, text, fresh=True)
                    tel = bot.telemetry()
                    if tel["l1_ok"]:
                        goal = plan.params.get("hand_goal")
                        note = f"queue={len(plan.queue)} skill={plan.skill} params={plan.params}"
                        if goal:
                            note += f" goal={goal}"
                    else:
                        note = "stand (L1 " + (tel.get("l1_err") or "offline") + ")"
                    await sock.send_json(
                        {"type": "log", "text": f"«{text}» → {plan.instruction} · {note}"}
                    )
                elif kind == "camera":
                    bot.orbit_camera(
                        float(raw.get("dx") or 0),
                        float(raw.get("dy") or 0),
                        float(raw.get("zoom") or 0),
                    )
                elif kind == "camera_reset":
                    bot.reset_demo_camera()
                elif kind == "reset":
                    await asyncio.to_thread(bot.reset_sim)
                    await sock.send_json({"type": "log", "text": "Reset · spawn."})
                elif kind == "auto_trial":
                    plan = bot.goal or bot.waypoint
                    await sock.send_json(
                        {"type": "log", "text": f"Auto-Trial · skill={plan.instruction[:40]}"}
                    )
                    result = await asyncio.to_thread(bot.auto_trial, plan)
                    for line in result.get("lines") or []:
                        await sock.send_json({"type": "log", "text": line})
                    await sock.send_json(
                        {
                            "type": "log",
                            "text": "Auto-Trial "
                            + ("success" if result.get("success") else "no-success")
                            + f" · {result.get('skill')}",
                        }
                    )
            elif bot.wants_verify():
                result = await _run_verify(bot)
                await sock.send_json(
                    {
                        "type": "log",
                        "text": f"verify done={result.get('done')} retry={result.get('retry')}",
                    }
                )
            elif bot.l1_due():
                plan = await _run_l1(bot, bot.user_cmd, fresh=False)
                if plan.done:
                    await sock.send_json({"type": "log", "text": "L1: done"})
            for note in bot.pop_notes():
                await sock.send_json({"type": "log", "text": note})
            await sock.send_json(bot.telemetry())
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        from tests.smoke import smoke

        smoke()
    else:
        import uvicorn

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            reload=False,
        )
