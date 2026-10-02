"""L1 VLM planner. Soft skill+params only; the foundation controller owns balance."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from agent.plan import Plan, attach_arm_channel
from agent.reach import attach_reach, strip_planner_pixels


def _pointing_rules() -> str:
    """Vision-only grasp pixel rules (plan_point / gaze)."""
    return (
        "Name a hand and the grasp place on the attached head image: "
        "u=0 left, u=1 right, v=0 top, v=1 bottom.\n"
        "Put u,v on the graspable surface the command names — what you see, not image center.\n"
        "Opening means the metal HANDLE you pull — not glass, knobs, burners, or flat panels.\n"
        "Do NOT point at walls, floor, seams, or borders. Keep 0.12 < u < 0.88.\n"
        "If the target is not clearly visible, reply {\"visible\": false}.\n"
    )


def _motion_plan_rules() -> str:
    """Main planner: motion and intent only — never u,v."""
    return (
        "Touch/grasp/open on a visible object: put hand_goal with ONLY "
        '{"hand":"left"|"right"|"both"} on the interaction frame — no u, no v. '
        "Pixels come from a separate pointing step.\n"
        "Robot head target: hand_goal {hand, target: head} only.\n"
    )


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
        self.model = os.getenv("L1_MODEL", "qwen3.5:9b-q8_0").strip()
        self.base_url = resolve_ollama_url()
        self.last_err = ""
        print(f"L1 planner model={self.model} url={self.base_url}", flush=True)

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
            queue = data.get("queue") if isinstance(data.get("queue"), list) else []
            plan = Plan(
                instruction=str(data.get("instruction") or user_command),
                skill=str(data.get("skill") or "hold"),
                params=params,
                done=bool(data.get("done", False)),
                queue=queue,
            )
            attach_reach(plan, user_command)
            attach_arm_channel(plan, user_command)
            strip_planner_pixels(plan)
            return plan
        except (TypeError, ValueError):
            return None

    async def plan(self, user_command: str, scene: dict, image: bytes = b"") -> tuple[Plan, bool]:
        prompt = (
            "You are the slow planner for a standing humanoid. The image is the camera.\n"
            "The foundation controller balances at control rate. If outcome is fall or failed: "
            'done=true, skill="hold", params={}. Do not invent a recovery gait.\n'
            "Reply JSON with keys instruction, queue, done. skill and params are optional.\n"
            "queue is an ordered list of command frames, at most 6. Each frame may set "
            "vx (m/s), wz, h_m (pelvis height in meters), hold_s (seconds), pose, hand, hands, "
            "direction, speed, depth. Do not pick a single skill name when the sentence has "
            "several parts: write them as successive frames. The body already balances.\n"
            "A frame without a skill is valid. vx, height and arms are one command.\n"
            "Putting a hand on the robot's own head is not a skill name. Write hand_goal "
            '{hand: left|right|both, target: head}. Do not answer hold.\n'
            '  "положи левую руку на голову" → {"instruction":"левая рука на голову","queue":['
            '{"hand_goal":{"hand":"left","target":"head"},"vx":0,"hold_s":4}],"done":false}\n'
            "Touching something visible is the same channel, for any object.\n"
            f"{_motion_plan_rules()}"
            '  object not in frame → {"instruction":"ищу объект","queue":['
            '{"direction":"left","hold_s":2},{"vx":0,"hold_s":1}],"done":false}\n'
            '  too close / cut off → {"instruction":"отойти","queue":['
            '{"direction":"backward","speed":"slow","hold_s":2},{"vx":0,"hold_s":1}],"done":false}\n'
            '  open the oven → {"instruction":"открыть духовку","queue":['
            '{"hand_goal":{"hand":"right"},"vx":0,"hold_s":8}],"done":false}\n'
            '  open the fridge → {"instruction":"открыть холодильник","queue":['
            '{"hand_goal":{"hand":"right"},"vx":0,"hold_s":8}],"done":false}\n'
            '  pick up the toaster → {"instruction":"поднять тостер","queue":['
            '{"hand_goal":{"hand":"both"},"vx":0,"hold_s":8}],"done":false}\n'
            "Examples:\n"
            '  "сделай 5 шагов вперед" → {"skill":"locomote","params":{"direction":"forward","speed":"medium","distance_hint":"5"}}\n'
            '  "иди вперёд" → {"skill":"locomote","params":{"direction":"forward","speed":"medium"}}\n'
            '  "стой" / "замри" → {"skill":"stand","params":{}}\n'
            '  "опусти руки" → {"skill":"hold","params":{"hands":"down"}}\n'
            '  "подними руки" → {"skill":"reach","params":{"hand":"both"}}\n'
            '  "подними правую руку" → {"skill":"reach","params":{"hand":"right"}}\n'
            '  "махни правой" → {"skill":"wave","params":{"hand":"right"}}\n'
            '  "руки в стороны" → {"skill":"reach","params":{"pose":"t"}}\n'
            '  "хлопни" → {"skill":"wave","params":{"pose":"clap"}}\n'
            '  "ударь правой ногой" → {"skill":"kick","params":{"foot":"right"}}\n'
            '  "присесть" / "наклонись" → {"skill":"squat","params":{"depth":"low"}}\n'
            '  "повернись налево" → {"skill":"turn","params":{"direction":"left"},"done":false}\n'
            '  "повернись направо" → {"skill":"turn","params":{"direction":"right"},"done":false}\n'
            '  "повернись на 90" → {"skill":"turn","params":{"direction":"left","angle":"90"},"done":false}\n'
            "One skill is the primary motion. Extra parts stay in params and run together "
            "(walk and arms, squat and arms). Do not drop the second part.\n"
            '  "иди вперед и руки в стороны" → {"skill":"locomote","params":{"direction":"forward","speed":"medium","pose":"t"}}\n'
            '  "присядь и подними руки" → {"skill":"squat","params":{"depth":"low","hand":"both"}}\n'
            '  "иди назад и подними правую руку" → {"skill":"locomote","params":{"direction":"backward","speed":"medium","hand":"right"}}\n'
            "Prefer queue when the command is a sequence:\n"
            '  "иди вперед, потом руки в стороны, потом стой" → {"instruction":"вперед, руки, стой","queue":['
            '{"direction":"forward","speed":"medium","hold_s":3},'
            '{"pose":"t","vx":0,"hold_s":2},'
            '{"vx":0,"hands":"down","hold_s":2}],"done":false}\n'
            '  "подними руки и иди вперед" → {"instruction":"руки и вперед","queue":['
            '{"hand":"both","vx":0.4,"hold_s":4}],"done":false}\n'
            "If the scene has requested_yaw and achieved_yaw: keep skill=turn and done=false until "
            "|achieved_yaw-requested_yaw| is small; then done=true, skill=hold.\n"
            f"Now: pelvis_z={scene.get('pelvis_z')} tilt={scene.get('tilt')} "
            f"outcome={scene.get('outcome')} skill={scene.get('skill')} "
            f"queue={scene.get('queue_i')}/{scene.get('queue_len')} ahead_m={scene.get('ahead_m')} "
            f"requested_yaw={scene.get('requested_yaw')} achieved_yaw={scene.get('achieved_yaw')} "
            f"done={scene.get('done')}\n"
            "ahead_m is clear space in front of the head camera. "
            "If queue is not yet on its last frame, keep the same queue and done=false.\n"
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
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 512, "think": False},
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

    async def plan_point(
        self, user_command: str, image: bytes, prior: dict | None = None
    ) -> tuple[Plan, bool]:
        """Relook with memory of the approach pixel. No lean."""
        prior_line = ""
        if prior and prior.get("u") is not None and prior.get("v") is not None:
            prior_line = (
                f"Earlier, from farther away, the grasp pixel was "
                f"u={float(prior['u']):.2f} v={float(prior['v']):.2f} "
                f"(hand={prior.get('hand', 'right')}). "
                "That was the intended handle/grip. "
                "If you still see that same handle, refine u,v onto it. "
                "If the handle is out of frame or occluded, reply {\"visible\": false} — "
                "do NOT invent a point on the flat door, wall, or image border.\n"
            )
        prompt = (
            "The image is the robot head camera, upright, often close to the object.\n"
            f"Command: {user_command}\n"
            f"{prior_line}"
            f"{_pointing_rules()}"
            "Reply JSON with keys hand, u, v — or visible:false."
        )
        message: dict = {"role": "user", "content": prompt, "images": [base64.b64encode(image).decode("ascii")]}
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 120, "think": False},
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                text = r.json()["message"]["content"]
            text = text.strip()
            if "</think>" in text:
                text = text.split("</think>", 1)[-1]
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1] if start >= 0 else text)
            if data.get("visible") is False or (data.get("u") is None and data.get("v") is None):
                # Keep gait as hold — engine restores the approach world point.
                plan = Plan(
                    instruction=user_command,
                    skill="hold",
                    params={},
                    done=False,
                    queue=[{"vx": 0, "hold_s": 4}],
                )
                print("gaze point: not visible up close -> keep prior", flush=True)
                self.last_err = ""
                return plan, True
            hand = str(data.get("hand") or (prior or {}).get("hand") or "right").strip().lower()
            if hand not in {"left", "right", "both"}:
                hand = "right"
            u = float(np_clip(data.get("u")))
            v = float(np_clip(data.get("v")))
            if u < 0.10 or u > 0.90:
                plan = Plan(
                    instruction=user_command,
                    skill="hold",
                    params={},
                    done=False,
                    queue=[{"vx": 0, "hold_s": 4}],
                )
                print(f"gaze point: border u={u:.2f} -> keep prior", flush=True)
                self.last_err = ""
                return plan, True
            goal = {"hand": hand, "u": u, "v": v}
            plan = Plan(
                instruction=user_command,
                skill="hold",
                params={"hand_goal": goal},
                done=False,
                queue=[{"hand_goal": goal, "vx": 0, "hold_s": 8}],
            )
            attach_reach(plan, user_command)
            print(f"gaze point hand={hand} u={u:.2f} v={v:.2f}", flush=True)
            self.last_err = ""
            return plan, True
        except Exception as e:
            self.last_err = f"{type(e).__name__}: {e}"[:160]
            print(f"gaze point failed: {self.last_err}", flush=True)
            return self.hold(user_command, {}), False

    async def verify(self, user_command: str, image: bytes) -> dict:
        """Did the language goal succeed in this head frame? Visual done, no joints."""
        prompt = (
            "The image is the robot head camera.\n"
            f"Command: {user_command}\n"
            "Has this command been successfully completed in the scene you see?\n"
            "Examples: a door/fridge/oven that should be open is clearly open; "
            "a toaster/appliance that should be on shows that state; "
            "an object that should be grasped is clearly in hand.\n"
            "If not done: is the graspable part (handle/button/object) still visible "
            "so the robot should point again?\n"
            'Reply JSON only: {"done": true|false, "retry": "hold"|"repoint"}.\n'
            'Use retry="repoint" when the target is visible but the task is not done. '
            'Use retry="hold" when waiting or the view is unclear.'
        )
        message: dict = {"role": "user", "content": prompt, "images": [base64.b64encode(image).decode("ascii")]}
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 80, "think": False},
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                text = r.json()["message"]["content"]
            text = text.strip()
            if "</think>" in text:
                text = text.split("</think>", 1)[-1]
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1] if start >= 0 else text)
            done = bool(data.get("done"))
            retry = str(data.get("retry") or ("hold" if done else "repoint")).strip().lower()
            if retry not in {"hold", "repoint", "look", "point"}:
                retry = "repoint" if not done else "hold"
            self.last_err = ""
            return {"done": done, "retry": retry}
        except Exception as e:
            self.last_err = f"{type(e).__name__}: {e}"[:160]
            print(f"verify failed: {self.last_err}", flush=True)
            return {"done": False, "retry": "hold"}


def np_clip(value) -> float:
    return max(0.0, min(1.0, float(value)))
