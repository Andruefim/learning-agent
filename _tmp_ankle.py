import os

os.environ["L2_DEVICE"] = "cpu"
from engine import RobotEngine, Plan, R_AZ, L_AZ, PIVOT_HX_SHIFT

bot = RobotEngine()
bot._student = lambda *a, **k: bot._last_teacher.copy()
print("shift", PIVOT_HX_SHIFT, flush=True)

def drive(plan, steps):
    max_az = 0.0
    n_brace = 0
    for i in range(steps):
        if i % 50 == 0:
            bot.apply_plan(plan, fresh=(i == 0), l1_ok=True)
        bot.step()
        if bot.outcome == "brace":
            n_brace += 1
        h = bot._hinges()
        max_az = max(max_az, abs(float(h[R_AZ])), abs(float(h[L_AZ])))
    return max_az, n_brace, bot.telemetry()

bot.reset_sim()
for _ in range(200):
    bot.step()
p90 = Plan("left90", skill="turn", params={"direction": "left", "angle": "90"})
az, nb, t1 = drive(p90, 1400)
print("c1", t1["achieved_yaw"], t1["outcome"], t1["pelvis_z"], "az", round(az, 3), "brace", nb, flush=True)
_, _, t2 = drive(p90, 1400)
print("c2", t2["achieved_yaw"], t2["outcome"], t2["pelvis_z"], t2["turn_mode"], flush=True)
for name, params in (("right", {"direction": "right"}), ("left", {"direction": "left"})):
    bot.reset_sim()
    for _ in range(200):
        bot.step()
    az, nb, t = drive(Plan(name, skill="turn", params=params), 1400)
    print(name, t["achieved_yaw"], t["outcome"], t["pelvis_z"], "az", round(az, 3), "brace", nb, flush=True)
bot.close()
