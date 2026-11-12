import numpy as np

import carsim as c


def test_no_drag_constant_acceleration():
    t, v = c.simulate(c.rk4_step, T=10, dt=0.01, drag=False, rolling=False)
    assert np.allclose(v, c.F / c.m * t)


def test_drag_only_matches_tanh():
    t, v = c.simulate(c.rk4_step, T=100, dt=0.05, rolling=False)
    assert np.allclose(v, c.v_tanh(t), atol=1e-8)


def test_terminal_speed_and_force_balance():
    t, v = c.simulate(c.rk4_step, T=600, dt=0.1)
    assert abs(v[-1] - c.vt_full()) < 1e-4
    residual = c.F - c.k * v[-1] ** 2 - c.f_roll
    assert abs(residual) < 1e-2          # newtons, essentially zero


def test_convergence_orders():
    dts = np.array([4, 2, 1, 0.5, 0.25])
    e_euler, e_rk4 = [], []
    for dt in dts:
        t, v = c.simulate(c.euler_step, T=64, dt=dt, rolling=False)
        e_euler.append(np.max(abs(v - c.v_tanh(t))))
        t, v = c.simulate(c.rk4_step, T=64, dt=dt, rolling=False)
        e_rk4.append(np.max(abs(v - c.v_tanh(t))))
    assert abs(c.order(dts, e_euler) - 1) < 0.2
    assert abs(c.order(dts, e_rk4) - 4) < 0.5


def test_power_at_100_kmh():
    v = 100 / 3.6
    assert 12000 < c.power(v) < 13000    # about 12.4 kW
