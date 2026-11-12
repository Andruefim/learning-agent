"""FastAPI server for Unitree G1 SLEEP Continual-Learning Assistant."""

from __future__ import annotations

import asyncio
import json
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
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform in ("darwin", "win32") else "egl"

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from agent import RobotEngine

engine: RobotEngine | None = None
_stop = threading.Event()
_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, _thread
    engine = RobotEngine(defer_render=True)
    _stop.clear()
    _thread = threading.Thread(target=engine.loop, args=(_stop,), daemon=True)
    _thread.start()
    yield
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2.0)
    if engine:
        engine.close()


app = FastAPI(title="Unitree G1 · SLEEP Continual Learning", lifespan=lifespan)


def _bot() -> RobotEngine:
    assert engine is not None
    return engine


@app.get("/")
def index():
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"ok": True, **_bot().telemetry()}


def _mjpeg(request: Request, get_frame):
    async def frames():
        last = None
        while not _stop.is_set():
            if await request.is_disconnected():
                break
            jpeg = await asyncio.to_thread(get_frame)
            if jpeg and jpeg is not last:
                last = jpeg
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            else:
                await asyncio.sleep(0.025)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@app.get("/video_feed")
def video_feed(request: Request):
    return _mjpeg(request, lambda: _bot().get_overview_jpeg())


@app.get("/eye_feed")
def eye_feed(request: Request):
    return _mjpeg(request, lambda: _bot().get_head_jpeg())


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    bot = _bot()

    async def sender():
        try:
            while True:
                for entry in bot.pop_logs():
                    await sock.send_json({"type": "log", **entry})
                await sock.send_json({"type": "telemetry", **bot.telemetry()})
                await asyncio.sleep(0.10)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    async def receiver():
        try:
            while True:
                raw_txt = await sock.receive_text()
                try:
                    msg = json.loads(raw_txt)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "command":
                    text = str(msg.get("text", "")).strip()
                    if text:
                        res = await asyncio.to_thread(bot.dispatch_command, text)
                        if not res.get("ok"):
                            await sock.send_json({"type": "log", "level": "error", "text": str(res.get("error"))})
                elif kind == "power":
                    bot.set_power(bool(msg.get("state", True)))
                elif kind == "standby":
                    bot.set_standby(bool(msg.get("state", False)))
                elif kind == "set_tau":
                    bot.set_tau_think(float(msg.get("tau", 0.35)))
                elif kind == "reset":
                    await asyncio.to_thread(bot.reset_sim)
                elif kind == "camera":
                    bot.orbit_camera(
                        float(msg.get("dx") or 0.0),
                        float(msg.get("dy") or 0.0),
                        float(msg.get("zoom") or 0.0),
                    )
                elif kind == "camera_reset":
                    bot.reset_camera()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    send_t = asyncio.create_task(sender())
    recv_t = asyncio.create_task(receiver())
    _done, pending = await asyncio.wait([send_t, recv_t], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()


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
