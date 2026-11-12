"""The car model: m dv/dt = F - 0.5*rho*Cd*A*v^2 - Crr*m*g"""
import numpy as np

# Parameters from the problem
m = 1200.0    # mass (kg)
F = 2000.0    # driving force (N)
rho = 1.2     # air density (kg/m^3)
Cd = 0.30     # drag coefficient
A = 2.2       # frontal area (m^2)
Crr = 0.012   # rolling resistance coefficient
g = 9.81      # gravity (m/s^2)

k = 0.5 * rho * Cd * A   # so that air drag = k * v^2
f_roll = Crr * m * g     # rolling resistance force (N), same at every speed


def accel(v, drag=True, rolling=True):
    """Acceleration dv/dt. Switch drag or rolling resistance off with False."""
    force = F
    if drag:
        force -= k * v**2
    if rolling:
        force -= f_roll
    return force / m


def euler_step(v, dt, **opts):
    return v + dt * accel(v, **opts)


def rk4_step(v, dt, **opts):
    k1 = accel(v, **opts)
    k2 = accel(v + 0.5 * dt * k1, **opts)
    k3 = accel(v + 0.5 * dt * k2, **opts)
    k4 = accel(v + dt * k3, **opts)
    return v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate(step, T, dt, **opts):
    """Start at v=0 and take steps of size dt until time T. Returns t, v arrays."""
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    v = np.zeros(n + 1)
    for i in range(n):
        v[i + 1] = step(v[i], dt, **opts)
    return t, v


# ---------- exact (analytic) results ----------
def vt_drag_only():
    """Terminal speed with Crr = 0: sqrt(2F / (rho*Cd*A))."""
    return np.sqrt(2 * F / (rho * Cd * A))


def v_tanh(t):
    """Exact solution with drag only: v = vt * tanh(t/tau), tau = m*vt/F."""
    vt = vt_drag_only()
    tau = m * vt / F
    return vt * np.tanh(t / tau)


def vt_full():
    """Terminal speed with drag AND rolling resistance."""
    return np.sqrt((F - f_roll) / k)


def power(v):
    """Power (W) needed to overcome air drag and rolling resistance at speed v."""
    return (k * v**2 + f_roll) * v


def order(dts, errors):
    """Order of convergence = slope of log(error) vs log(dt)."""
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]
