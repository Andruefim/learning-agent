"""L1 VLM planner. Soft skill+params only; the foundation controller owns balance."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from agent.plan import Plan


def _wsl_windows_host() -> str | None:
    try:
        ver = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return None
    if "microsoft" not in ver and "wsl" not in ver:
        return None
    try:
        with open("/proc/net/route", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "00000000":
                    gw = int(parts[2], 16)
                    return f"{gw & 0xFF}.{(gw >> 8) & 0xFF}.{(gw >> 16) & 0xFF}.{(gw >> 24) & 0xFF}"
    except OSError:
        return None
    return None


def resolve_ollama_url(spec: str | None = None) -> str:
    url = (spec or os.getenv("L1_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    if host not in {"127.0.0.1", "localhost"}:
        return url
    win = _wsl_windows_host()
    if not win:
        return url
    scheme, _, rest = url.partition("://")
    _, sep, after = rest.partition(":")
    return f"{scheme}://{win}:{after}" if sep else f"{scheme}://{win}"


class Level1Planner:
    def __init__(self):
        self.model = os.getenv("L1_MODEL", "qwen3.8:latest").strip()
        self.base_url = resolve_ollama_url()
        self.last_err = ""

    def hold(self, user_command: str, scene: dict) -> Plan:
        return Plan.stand(user_command.strip() or "stand")

    def _parse(self, text: str, user_command: str) -> Plan | None:
        text = text.strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict):
            return None
        try:
            params = data.get("params") if isinstance(data.get("params"), dict) else {}
            return Plan(
                instruction=str(data.get("instruction") or user_command),
                skill=str(data.get("skill") or "hold"),
                params=params,
                done=bool(data.get("done", False)),
            )
        except (TypeError, ValueError):
            return None

    async def plan(self, user_command: str, scene: dict, image: bytes = b"") -> tuple[Plan, bool]:
        prompt = (
            "You are the slow planner for a standing humanoid. The image is the camera.\n"
            "The foundation controller balances at control rate. If outcome is fall or failed: "
            'done=true, skill="hold", params={}. Do not invent a recovery gait.\n'
            "Reply JSON with keys instruction, skill, params, done.\n"
            "skill is one of: hold, stand, squat, locomote, turn, wave, kick, reach.\n"
            "params is a SOFT dictionary of words, not motor ticks. No absolute coordinates, "
            "no object names, no cube. L2 turns skill+params into motion.\n"
            "Examples:\n"
            '  "сделай 5 шагов вперед" → {"skill":"locomote","params":{"direction":"forward","speed":"medium","distance_hint":"5"}}\n'
            '  "иди вперёд" → {"skill":"locomote","params":{"direction":"forward","speed":"medium"}}\n'
            '  "стой" / "замри" → {"skill":"stand","params":{}}\n'
            '  "опусти руки" → {"skill":"hold","params":{"hands":"down"}}\n'
            '  "подними правую руку" → {"skill":"reach","params":{"hand":"right"}}\n'
            '  "махни правой" → {"skill":"wave","params":{"hand":"right"}}\n'
            '  "руки в стороны" → {"skill":"reach","params":{"pose":"t"}}\n'
            '  "хлопни" → {"skill":"wave","params":{"pose":"clap"}}\n'
            '  "ударь правой ногой" → {"skill":"kick","params":{"foot":"right"}}\n'
            '  "присесть" / "наклонись" → {"skill":"squat","params":{"depth":"low"}}\n'
            '  "повернись налево" → {"skill":"turn","params":{"direction":"left"},"done":false}\n'
            '  "повернись направо" → {"skill":"turn","params":{"direction":"right"},"done":false}\n'
            '  "повернись на 90" → {"skill":"turn","params":{"direction":"left","angle":"90"},"done":false}\n'
            "If the scene has requested_yaw and achieved_yaw: keep skill=turn and done=false until "
            "|achieved_yaw-requested_yaw| is small; then done=true, skill=hold.\n"
            f"Now: pelvis_z={scene.get('pelvis_z')} tilt={scene.get('tilt')} "
            f"outcome={scene.get('outcome')} skill={scene.get('skill')} "
            f"requested_yaw={scene.get('requested_yaw')} achieved_yaw={scene.get('achieved_yaw')} "
            f"done={scene.get('done')}\n"
            f"User command: {user_command}\n"
            "JSON only. instruction is a short paraphrase of THIS command."
        )
        message: dict = {"role": "user", "content": prompt}
        if image:
            message["images"] = [base64.b64encode(image).decode("ascii")]
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 256, "think": False},
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                text = r.json()["message"]["content"]
            parsed = self._parse(text, user_command)
            if parsed is None:
                self.last_err = "bad json from " + self.model
                return self.hold(user_command, scene), False
            self.last_err = ""
            return parsed, True
        except Exception as e:
            self.last_err = f"{type(e).__name__}: {e}"[:160]
            return self.hold(user_command, scene), False
