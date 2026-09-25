"""Drive one command and record pose from /health while the planner runs."""

import asyncio
import json
import threading
import time
import urllib.request

import websockets

OUT = r"C:\Users\Andrey\Desktop\agi\learning-agent\tests"
STOP = False


def health() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def grab(path: str) -> None:
    req = urllib.request.Request("http://127.0.0.1:8000/video_feed")
    with urllib.request.urlopen(req, timeout=8) as resp:
        blob = b""
        while b"\xff\xd9" not in blob:
            chunk = resp.read(65536)
            if not chunk:
                break
            blob += chunk
            if len(blob) > 8_000_000:
                break
    start = blob.find(b"\xff\xd8")
    end = blob.find(b"\xff\xd9", start)
    open(path, "wb").write(blob[start : end + 2])


def poll() -> None:
    t0 = time.time()
    last = None
    saved = set()
    while not STOP and time.time() - t0 < 55:
        try:
            msg = health()
        except Exception as exc:
            print("health", exc, flush=True)
            time.sleep(0.4)
            continue
        now = time.time() - t0
        row = (
            f"{now:5.1f} z={msg.get('pelvis_z')} tilt={msg.get('tilt')} x={msg.get('x')} "
            f"vx={msg.get('vx')} out={msg.get('outcome')} ctrl={msg.get('ctrl')} "
            f"status={msg.get('status')}"
        )
        if row != last:
            print(row, flush=True)
            last = row
        mark = int(now)
        if mark in (10, 18, 26, 34, 42) and mark not in saved and msg.get("outcome") != "fall":
            saved.add(mark)
            try:
                grab(OUT + rf"\_live_{mark}.jpg")
                print("FRAME", mark, flush=True)
            except Exception as exc:
                print("frame", exc, flush=True)
        if msg.get("outcome") == "fall" and "fall" not in saved and now > 5:
            saved.add("fall")
            try:
                grab(OUT + r"\_live_fall.jpg")
                print("FRAME fall", flush=True)
            except Exception as exc:
                print("frame", exc, flush=True)
        time.sleep(0.3)


async def main() -> None:
    global STOP
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=2**22) as ws:
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.5)
        print("SEND command", flush=True)
        await ws.send(json.dumps({"type": "command", "text": "\u043f\u043e\u0434\u043d\u0438\u043c\u0438 \u0442\u043e\u0441\u0442\u0435\u0440"}))
        t1 = time.time()
        while time.time() - t1 < 50:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=25)
            except TimeoutError:
                print("ws timeout", flush=True)
                break
            msg = json.loads(raw)
            if msg.get("type") == "log":
                open(OUT + r"\_watch_log.txt", "a", encoding="utf-8").write(msg.get("text", "") + "\n")
                print("LOG", len(msg.get("text", "")), flush=True)
    STOP = True
    thread.join(timeout=2)


if __name__ == "__main__":
    asyncio.run(main())
