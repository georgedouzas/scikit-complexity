"""
# Finite-time blow-up

Hyperbolic growth `x' = x²` looks tame at first, slower than an exponential, but every doubling takes half the time of
the previous one, and the state diverges at the finite time `t* = 1 / x0`: a singularity reached in finite time.
"""

# %%
# The run starts at `x = 1` with `p = 2`, so the singularity is at `t* = 1`, and stops at `t = 0.99`, where the exact
# solution `x = 1 / (1 - t)` reaches 100. The default blueprint shows `x` as a time series.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import FiniteTimeBlowUpSystem

SYSTEM = 'finite_time_blow_up'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(FiniteTimeBlowUpSystem(), until=0.99, dt=0.001, observers=[observer])

# %%
# Approach to the singularity
# ---------------------------
# The exact solution `x = ((p - 1)(t* - t))^(-1 / (p - 1))` diverges as a power of the time left, with
# `t* = 1 / (p - 1)` for `x0 = 1` and `a = 1`. Plotted against `t* - t` on log-log axes, runs for `p = 1.5`, 2, and 3
# should follow straight lines of slopes -2, -1, and -1/2.

fig, ax = plt.subplots(figsize=(6, 4.5))
for p in (1.5, 2.0, 3.0):
    singular = 1 / (p - 1)
    trajectory = simulate(FiniteTimeBlowUpSystem(p=p), until=0.999 * singular, dt=singular / 2000)
    remaining = singular - trajectory.times
    (line,) = ax.loglog(remaining, trajectory.values('growth.x'), label=f'p = {p}')
    ax.loglog(remaining, ((p - 1) * remaining) ** (-1 / (p - 1)), ':', color='black')
ax.set_xlabel(r'time left to the singularity $t^* - t$')
ax.set_ylabel('x')
ax.set_title('Power-law divergence (dotted: exact solutions)')
ax.legend()
plt.show()
