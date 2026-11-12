"""System 2 Vision-Language Planner: Gemini Cloud primary + automatic Ollama fallback."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np
from PIL import Image

PRIMITIVES = ["press_down", "push_forward", "pull_back", "lift_up", "slide_left", "slide_right"]

PRIMITIVE_HELP = (
    "press_down = push downward (toaster lever, top button); "
    "push_forward = push away from robot (close door/drawer, press front button/panel); "
    "pull_back = hook/grasp handle and pull toward robot (open oven, fridge, dishwasher, cabinet, drawer); "
    "lift_up = lift handle/lid upward; "
    "slide_left / slide_right = swing or slide sideways (e.g. side-hinged fridge/cabinet handle)."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 86) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    txt = text.strip()
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[-1].strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = _JSON_RE.search(txt)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _norm_point_to_uv(pt: Any, width: int, height: int) -> Optional[tuple[float, float]]:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        u = x * width
        v = y * height
    else:
        u = float(np.clip(x, 0.0, 1000.0)) / 1000.0 * width
        v = float(np.clip(y, 0.0, 1000.0)) / 1000.0 * height
    return float(np.clip(u, 0.0, width - 1.0)), float(np.clip(v, 0.0, height - 1.0))


class VisionLanguagePlanner:
    """Deliberative VLM planner using Cloud Gemini first and falling back to local Ollama."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.ollama_model = os.getenv("L1_MODEL", "qwen3.5:9b").strip()
        self.ollama_url = resolve_ollama_url()
        self.log = log
        self.active_backend = "gemini" if self.api_key else "ollama"
        self.last_error = ""

    def _call_gemini(self, prompt: str, images: List[np.ndarray], max_tokens: int) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        parts: list[dict] = [{"text": prompt}]
        for img in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": _encode_jpeg_b64(img),
                    }
                }
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        with httpx.Client(timeout=20.0) as client:
            r = client.post(url, headers={"x-goog-api-key": self.api_key}, json=body)
            if r.status_code == 400 and "thinkingConfig" in body["generationConfig"]:
                body["generationConfig"].pop("thinkingConfig", None)
                r = client.post(url, headers={"x-goog-api-key": self.api_key}, json=body)
            r.raise_for_status()
            data = r.json()
        texts = []
        for part in data["candidates"][0]["content"].get("parts") or []:
            if part.get("thought"):
                continue
            if part.get("text"):
                texts.append(part["text"])
        raw = "".join(texts)
        parsed = _extract_json(raw)
        if parsed is None:
            raise ValueError(f"Invalid JSON from Gemini: {raw[:120]}")
        return parsed

    def _call_ollama(self, prompt: str, images: List[np.ndarray], max_tokens: int) -> Dict[str, Any]:
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [_encode_jpeg_b64(img) for img in images]
        payload = {
            "model": self.ollama_model,
            "messages": [msg],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": max_tokens, "think": False},
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{self.ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            txt = r.json()["message"]["content"]
        parsed = _extract_json(txt)
        if parsed is None:
            raise ValueError(f"Invalid JSON from Ollama: {txt[:120]}")
        return parsed

    def _ask_json(self, prompt: str, images: List[np.ndarray], max_tokens: int = 384) -> Optional[Dict[str, Any]]:
        if self.api_key:
            try:
                res = self._call_gemini(prompt, images, max_tokens)
                self.active_backend = f"gemini ({self.gemini_model})"
                self.last_error = ""
                return res
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"[:160]
                self.last_error = err_msg
                self.log(f"[VLM] Gemini не ответил ({err_msg}) -> fallback на Ollama ({self.ollama_model})")

        try:
            res = self._call_ollama(prompt, images, max_tokens)
            self.active_backend = f"ollama ({self.ollama_model})"
            self.last_error = ""
            return res
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"[:160]
            self.last_error = err_msg
            self.log(f"[VLM Error] Ollama также недоступна ({err_msg})")
            return None

    @staticmethod
    def _lessons_block(lessons: Optional[List[str]]) -> str:
        if not lessons:
            return ""
        items = "\n".join(f"  - {str(x).strip()}" for x in lessons[-4:] if str(x).strip())
        if not items:
            return ""
        return (
            "\nEpisodic lessons from prior FAILED attempts on this exact command "
            "(avoid repeating these mistakes — adjust target point or motion):\n"
            f"{items}\n"
        )

    def plan_command(self, rgb: np.ndarray, command: str, *, lessons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify a natural-language command into body skill OR kitchen object manipulation + grounding."""
        lesson_txt = self._lessons_block(lessons)
        prompt = (
            "You control a standing humanoid robot (Unitree G1) in a kitchen. The image is its head camera.\n"
            f'Operator command: "{command}"\n'
            f"{lesson_txt}"
            "Decide whether the command is:\n"
            '  1) "body_skill": locomotion, turning, squatting, raising/waving arms, or holding pose (no kitchen object interaction).\n'
            '  2) "manipulate": interacting with a kitchen object/appliance (open/close oven, fridge, microwave, cabinet, drawer, dishwasher, press toaster/coffee button, etc.).\n'
            "If manipulate:\n"
            "  - Find the exact graspable part (handle, lever, button, door edge; NEVER walls or floor).\n"
            "  - Return point=[x, y] in 0..1000 image coordinates (x right, y down), or visible=false if not clearly visible.\n"
            f"  - Rank 1..4 candidate motions from {PRIMITIVES}: {PRIMITIVE_HELP}\n"
            '  - If visible=false, suggest search_dir="left" or "right" where the target likely is.\n'
            "If body_skill:\n"
            '  - Fill skill ("locomote"|"turn"|"squat"|"reach"|"wave"|"stand") and params '
            '({"direction":"forward|backward|left|right","steps":N,"angle":deg,"depth":"low","hand":"left|right|both","pose":"t|up|down"}).\n'
            "Reply JSON ONLY:\n"
            '{\n'
            '  "mode": "manipulate" | "body_skill",\n'
            '  "skill": "<if body_skill>",\n'
            '  "params": {},\n'
            '  "visible": true | false,\n'
            '  "object": "<name>",\n'
            '  "part": "<handle/lever/button>",\n'
            '  "point": [x, y],\n'
            '  "motions": ["pull_back"],\n'
            '  "hand": "right" | "left",\n'
            '  "search_dir": "left" | "right"\n'
            '}'
        )
        res = self._ask_json(prompt, [rgb], max_tokens=320) or {}
        mode = str(res.get("mode") or "manipulate").strip().lower()
        h, w = rgb.shape[:2]

        if mode == "body_skill":
            return {
                "mode": "body_skill",
                "skill": str(res.get("skill") or "stand").strip().lower(),
                "params": res.get("params") if isinstance(res.get("params"), dict) else {},
                "ok": bool(res),
            }

        uv = _norm_point_to_uv(res.get("point"), w, h)
        visible = bool(res.get("visible", False)) and (uv is not None)
        mots = [m for m in (res.get("motions") or []) if m in PRIMITIVES]
        for fallback in ("pull_back", "push_forward", "press_down", "lift_up"):
            if fallback not in mots:
                mots.append(fallback)
        search_dir = str(res.get("search_dir") or "right").strip().lower()
        if search_dir not in ("left", "right"):
            search_dir = "right"
        return {
            "mode": "manipulate",
            "visible": visible,
            "uv": uv,
            "object": str(res.get("object") or ""),
            "part": str(res.get("part") or ""),
            "motions": mots,
            "hand": "left" if str(res.get("hand") or "right").strip().lower() == "left" else "right",
            "search_dir": search_dir,
            "ok": bool(res),
        }

    def ground(self, rgb: np.ndarray, command: str, *, lessons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dedicated visual re-grounding up close after the robot walks to the appliance."""
        lesson_txt = self._lessons_block(lessons)
        prompt = (
            "This is the robot head camera, now standing close to the kitchen fixture.\n"
            f'Operator command: "{command}"\n'
            f"{lesson_txt}"
            "Point precisely at the graspable part (handle bar, lever, button, door edge) needed for the command.\n"
            "Do NOT point at flat doors without handle, floor, or walls.\n"
            f"Also rank candidate motions from {PRIMITIVES}: {PRIMITIVE_HELP}\n"
            "Reply JSON ONLY:\n"
            '{"visible": true|false, "object": "<name>", "part": "<part>", "point": [x, y], "motions": ["..."]}\n'
            "point uses 0..1000 image coordinates (x right, y down)."
        )
        res = self._ask_json(prompt, [rgb], max_tokens=200) or {}
        h, w = rgb.shape[:2]
        uv = _norm_point_to_uv(res.get("point"), w, h)
        visible = bool(res.get("visible", False)) and (uv is not None)
        mots = [m for m in (res.get("motions") or []) if m in PRIMITIVES]
        for fallback in ("pull_back", "push_forward", "press_down", "lift_up"):
            if fallback not in mots:
                mots.append(fallback)
        return {
            "visible": visible,
            "uv": uv,
            "object": str(res.get("object") or ""),
            "part": str(res.get("part") or ""),
            "motions": mots,
            "ok": bool(res),
        }

    def judge(self, before: np.ndarray, after: np.ndarray, command: str) -> Dict[str, Any]:
        """Visual verification: compare before vs after head camera frames."""
        prompt = (
            "Two head-camera images of the same robot task: FIRST = before, SECOND = after.\n"
            f'Operator command: "{command}"\n'
            "Did the physical scene visibly change so the command is now accomplished "
            "(e.g. door/drawer opened or closed, lever pressed down, button pushed)?\n"
            'Reply JSON ONLY: {"changed": true|false, "done": true|false, "what": "<short description>"}'
        )
        res = self._ask_json(prompt, [before, after], max_tokens=140) or {}
        return {
            "done": bool(res.get("done", False)),
            "changed": bool(res.get("changed", False)),
            "what": str(res.get("what") or res.get("what_changed") or ""),
            "ok": bool(res),
        }

    def reflect_failure(
        self,
        before: Optional[np.ndarray],
        after: Optional[np.ndarray],
        command: str,
        info: Dict[str, Any],
    ) -> str:
        """Reflect on a failed episode to produce a concise, non-heuristic episodic lesson."""
        tried = ", ".join(info.get("tried_motions") or []) or "none"
        ik_cm = float(info.get("ik_err_cm") or 0.0)
        fell = bool(info.get("fell", False))
        pt = info.get("target_pt")
        pt_str = f"[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]" if pt is not None else "unknown"
        what = str(info.get("what") or info.get("reason") or "no visible change")

        imgs = [im for im in (before, after) if im is not None]
        if imgs:
            prompt = (
                f'The humanoid robot FAILED to complete command: "{command}".\n'
                f"Execution telemetry: target_3d={pt_str}, IK_error={ik_cm:.1f}cm, "
                f"tried_motions=[{tried}], fell={fell}, outcome='{what}'.\n"
                "Inspect the camera image(s) (before/after if 2 images). Give ONE short concrete lesson "
                "(max 25 words) for the next retry explaining what went wrong (e.g., missed handle height/position, "
                "out-of-reach point causing fall, or wrong motion direction) and what to adjust.\n"
                'Reply JSON ONLY: {"lesson": "<short lesson in Russian or English>"}'
            )
            res = self._ask_json(prompt, imgs, max_tokens=120) or {}
            lesson = str(res.get("lesson") or "").strip()
            if lesson:
                return lesson[:220]

        parts = [f"прошлая попытка ({tried}) не удалась: {what}"]
        if ik_cm > 12.0:
            parts.append(f"точка цели была труднодосягаема (IK err={ik_cm:.0f}см) — выбери ручку на удобной высоте")
        if fell:
            parts.append("робот потерял равновесие — скорректируй точку и направление")
        return "; ".join(parts)[:220]
