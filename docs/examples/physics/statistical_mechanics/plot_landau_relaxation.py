"""
# Landau relaxation

In Landau's theory the order parameter of a magnet slides down the free energy
`F = a (T - Tc) phi^2 / 2 + b phi^4 / 4`. Below the critical temperature the free energy has two minima at
`phi = +-sqrt(a (Tc - T) / b)` and a small initial magnetization grows to one of them; above it the only minimum is
zero. The view shows the order parameter at `T = 0.5 Tc` growing from 0.1 along an S-shaped curve to `sqrt(0.5)`.
"""

# %%
# The model is created by name from the catalog with `a = b = Gamma = 1` and `Tc = 1`. The observer saves every frame
# of 0.1 with the default blueprint, whose primary view is the time series of the order parameter.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'landau_relaxation'

system = create(MODEL, T=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Critical slowing down
# ---------------------
# Linearizing the relaxation about the minimum gives an exponential approach with the time `1 / (a (T - Tc))` above
# `Tc` and `1 / (2 a (Tc - T))` below it, both diverging at `Tc`. The relaxation time measured from the late decay of
# `|phi - phi_min|` should follow these two branches on either side of the critical temperature.


def relaxation_time(temperature):
    """Return the relaxation time from the late exponential decay of the distance to the minimum."""
    until = 10 / abs(temperature - 1)
    run = Simulation(create(MODEL, T=temperature), dt=until / 400).run(until=until)
    phi = run.trajectory_['magnet.phi']
    target = np.sqrt(max(1 - temperature, 0))
    distance = np.abs(phi - target)
    late = slice(120, 240)
    slope = np.polyfit(run.trajectory_.times[late], np.log(distance[late]), 1)[0]
    return -1 / slope


temperatures = np.concatenate([np.linspace(0.5, 0.95, 8), np.linspace(1.05, 1.5, 8)])
times = [relaxation_time(temperature) for temperature in temperatures]
below = np.linspace(0.5, 0.97, 100)
above = np.linspace(1.03, 1.5, 100)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(temperatures, times, 'o', label='simulation')
ax.semilogy(below, 1 / (2 * (1 - below)), color='black', label='1 / (2 a (Tc - T))')
ax.semilogy(above, 1 / (above - 1), color='gray', label='1 / (a (T - Tc))')
ax.axvline(1.0, color='lightgray', linestyle='--')
ax.set_xlabel('temperature T / Tc')
ax.set_ylabel('relaxation time')
ax.set_title('The relaxation time diverges at the critical temperature')
ax.legend()
fig.tight_layout()
plt.show()
