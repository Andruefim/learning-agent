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


@app.get("/video_feed")
def video_feed():
    def frames():
        while True:
            jpeg = _bot().jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


async def _run_l1(bot: RobotEngine, text: str, *, fresh: bool) -> Plan:
    while not bot.begin_l1():
        await asyncio.sleep(0.05)
    cmd = text.strip()
    try:
        scene = bot.scene_brief()
        plan, ok = await bot.planner.plan(cmd, scene, bot.jpeg())
        if bot.intent == cmd:
            bot.apply_plan(plan, fresh=fresh, l1_ok=ok)
        return plan
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
                    plan = await _run_l1(bot, text, fresh=True)
                    tel = bot.telemetry()
                    if tel["l1_ok"]:
                        note = f"skill={plan.skill} params={plan.params}"
                    else:
                        note = "stand (L1 " + (tel.get("l1_err") or "offline") + ")"
                    await sock.send_json(
                        {"type": "log", "text": f"«{text}» → {plan.instruction} · {note}"}
                    )
                elif kind == "consolidate":
                    await sock.send_json({"type": "log", "text": "Save: training flow on replay…"})
                    msg = await asyncio.to_thread(bot.consolidate)
                    await sock.send_json({"type": "log", "text": msg})
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
            elif bot.l1_due():
                plan = await _run_l1(bot, bot.user_cmd, fresh=False)
                if plan.done:
                    await sock.send_json({"type": "log", "text": "L1: done"})
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
