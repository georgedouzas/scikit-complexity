"""
# Pendulum

A bob on a rigid rod swings back and forth under gravity. Small swings keep the period `2π sqrt(l / g)` whatever their
amplitude, but large swings take longer, and the period diverges as the amplitude approaches the upside-down position.
"""

# %%
# The pendulum is released from rest at 2.5 radians, a large swing. The default blueprint draws the rod and the bob,
# with the fading trace of the bob.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pendulum'
system = create(MODEL, theta=2.5, omega=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=5)

# %%
# Period against amplitude
# ------------------------
# The exact period of a pendulum released from rest at `θ0` is `T = 4 K(sin^2(θ0 / 2)) sqrt(l / g)`, with `K` the
# complete elliptic integral of the first kind. The periods measured from the upward zero crossings of the angular
# velocity should lie on this curve, which starts at `2π sqrt(l / g)` and diverges at `θ0 = π`.


def measured_period(amplitude):
    """Return the mean period of a pendulum released from rest at an amplitude."""
    run = Simulation(create(MODEL, theta=amplitude, omega=0.0), dt=0.005).run(until=30)
    times, velocity = np.asarray(run.trajectory_.times), np.ravel(run.trajectory_['bob.theta_d1'])
    crossings = times[1:][(velocity[:-1] < 0) & (velocity[1:] >= 0)]
    return np.diff(crossings).mean()


amplitudes = np.linspace(0.2, 3.0, 12)
grid = np.linspace(0.01, 3.1, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, 4 * ellipk(np.sin(grid / 2) ** 2) / np.sqrt(9.81), color='black', label='exact period')
ax.plot(amplitudes, [measured_period(amplitude) for amplitude in amplitudes], 'o', label='simulated')
ax.axhline(2 * np.pi / np.sqrt(9.81), linestyle='--', color='grey', label='small swings')
ax.set_xlabel('amplitude θ0 (radians)')
ax.set_ylabel('period')
ax.set_title('Large swings take longer')
ax.legend()
plt.show()
