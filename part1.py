"""Part 1: run all the experiments and make the plots."""
import os

import matplotlib.pyplot as plt
import numpy as np

import carsim as c

os.makedirs("figures", exist_ok=True)

# ---- 1. No drag: acceleration should be constant = F/m ----
t, v = c.simulate(c.rk4_step, T=20, dt=0.01, drag=False, rolling=False)
a_fit = np.polyfit(t, v, 1)[0]   # slope of v vs t
print("NO DRAG")
print(f"  F/m = {c.F / c.m:.6f},  simulated acceleration = {a_fit:.6f} m/s^2\n")

# ---- 2. Drag only: compare with v = vt*tanh(t/tau) ----
t, v_euler = c.simulate(c.euler_step, T=300, dt=0.05, rolling=False)
t, v_rk4 = c.simulate(c.rk4_step, T=300, dt=0.05, rolling=False)
print("DRAG ONLY")
print(f"  vt formula = {c.vt_drag_only():.4f} m/s")
print(f"  max error vs tanh:  Euler = {np.max(abs(v_euler - c.v_tanh(t))):.2e}, "
      f"RK4 = {np.max(abs(v_rk4 - c.v_tanh(t))):.2e}\n")

# ---- 3. Terminal speed and force balance ----
# (a) drag only
t, v = c.simulate(c.rk4_step, T=600, dt=0.1, rolling=False)
print("TERMINAL SPEED (drag only)")
print(f"  simulated = {v[-1]:.6f} m/s, formula = {c.vt_drag_only():.6f} m/s")
# (b) drag + rolling resistance
t, v = c.simulate(c.rk4_step, T=600, dt=0.1)
v_end = v[-1]
drag_end = c.k * v_end**2
print("TERMINAL SPEED (drag + rolling)")
print(f"  simulated = {v_end:.6f} m/s, formula = {c.vt_full():.6f} m/s")
print(f"  forces: F = {c.F:.2f} N, air drag = {drag_end:.2f} N, rolling = {c.f_roll:.2f} N")
print(f"  F - drag - rolling = {c.F - drag_end - c.f_roll:.2e} N (should be ~0)\n")

# ---- 4. Convergence: error vs step size (drag only, T = 64 s) ----
dts = np.array([4, 2, 1, 0.5, 0.25, 0.125])
err_euler, err_rk4 = [], []
for dt in dts:
    t, v = c.simulate(c.euler_step, T=64, dt=dt, rolling=False)
    err_euler.append(np.max(abs(v - c.v_tanh(t))))
    t, v = c.simulate(c.rk4_step, T=64, dt=dt, rolling=False)
    err_rk4.append(np.max(abs(v - c.v_tanh(t))))

order_euler = c.order(dts, err_euler)
order_rk4 = c.order(dts, err_rk4)
print("CONVERGENCE")
print("      dt        Euler          RK4")
for dt, e1, e2 in zip(dts, err_euler, err_rk4):
    print(f"  {dt:6.3f}  {e1:12.3e}  {e2:12.3e}")
print(f"  measured order: Euler = {order_euler:.2f} (expect 1), "
      f"RK4 = {order_rk4:.2f} (expect 4)\n")

# ---- 5. Power at 100 km/h ----
v100 = 100 / 3.6   # m/s
air = c.k * v100**2
P = c.power(v100)
print("POWER AT 100 km/h")
print(f"  air drag = {air:.1f} N, rolling = {c.f_roll:.1f} N")
print(f"  P = {P:.0f} W = {P / 1000:.2f} kW")
print(f"  share: air {100 * air / (air + c.f_roll):.0f}%, "
      f"rolling {100 * c.f_roll / (air + c.f_roll):.0f}%")

# ---- Plots ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(t_ := np.arange(0, 300.05, 0.05), c.v_tanh(t_), "k-", lw=3, alpha=0.4, label="exact tanh")
ax1.plot(t_, v_euler, "r--", label="Euler")
ax1.plot(t_, v_rk4, "b:", label="RK4")
ax1.set(xlabel="t (s)", ylabel="v (m/s)", title="Drag only: v(t)")
ax1.legend()

ax2.loglog(dts, err_euler, "ro-", label=f"Euler (slope {order_euler:.2f})")
ax2.loglog(dts, err_rk4, "bs-", label=f"RK4 (slope {order_rk4:.2f})")
ax2.set(xlabel="step size dt (s)", ylabel="max error (m/s)", title="Convergence")
ax2.legend()

plt.tight_layout()
plt.savefig("figures/part1.png", dpi=200)
plt.show()
