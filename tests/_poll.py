import json
import time
import urllib.request

t = time.time()
last = None
while time.time() - t < 18:
    raw = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).read().decode("utf-8")
    m = json.loads(raw)
    row = (
        f"{time.time() - t:5.1f} z={m['pelvis_z']} tilt={m['tilt']} x={m['x']} "
        f"vx={m['vx']} out={m['outcome']} ctrl={m['ctrl']} yaw={m['yaw']} "
        f"r={m['r_arm']} l={m['l_arm']}"
    )
    if row != last:
        print(row, flush=True)
        last = row
    time.sleep(0.3)
