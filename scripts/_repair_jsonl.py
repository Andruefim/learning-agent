"""One-shot: drop bad/dupe lines from a Tier B jsonl."""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    p = pathlib.Path(sys.argv[1])
    rows: list[dict] = []
    bad = 0
    seen: set[int] = set()
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            bad += 1
            print(f"bad line {i}: {e} :: {line[:80]!r}")
            continue
        seed = obj.get("seed")
        if seed in seen:
            print(f"dupe seed {seed} at line {i}, keep last")
            rows = [r for r in rows if r.get("seed") != seed]
        seen.add(seed)
        rows.append(obj)
    rows.sort(key=lambda r: r["seed"])
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ok = sum(1 for r in rows if r.get("success"))
    print(f"kept {len(rows)} clean rows, bad={bad}, success={ok}")


if __name__ == "__main__":
    main()
