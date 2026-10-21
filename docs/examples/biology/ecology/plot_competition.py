"""
# Competition

Two species that compete for the same resources each grow towards their own carrying capacity, but every member of
one species takes a share of the resources of the other. When each species limits itself more than it limits its
rival, both settle below their capacities and coexist.
"""

# %%
# The model is created by name from the catalog with its default coefficients, which satisfy the condition of
# coexistence `alpha12 alpha21 < 1`. The observer saves every second frame with the default blueprint, whose primary
# view is the time series of both populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'competition'

system = create(MODEL, first=5.0, second=60.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=30)

# %%
# Phase plane and nullclines
# --------------------------
# The first species stops growing on the line `n1 + alpha12 n2 = K1` and the second on `alpha21 n1 + n2 = K2`. The
# lines cross at the coexistence equilibrium, and the trajectory should run into that crossing.

first, second = simulation.trajectory_['first.n'], simulation.trajectory_['second.n']
K1, K2, alpha12, alpha21 = 100.0, 80.0, 0.5, 0.6
n1 = (K1 - alpha12 * K2) / (1 - alpha12 * alpha21)
n2 = (K2 - alpha21 * K1) / (1 - alpha12 * alpha21)
grid = np.linspace(0, K1, 100)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(grid, (K1 - grid) / alpha12, color='tab:blue', linestyle='--', label='first stops growing')
ax.plot(grid, K2 - alpha21 * grid, color='tab:orange', linestyle='--', label='second stops growing')
ax.plot(first, second, color='black', label='trajectory')
ax.plot(n1, n2, 'o', color='black', label=f'coexistence ({n1:.1f}, {n2:.1f})')
ax.set_xlim(0, K1 * 1.05)
ax.set_ylim(0, K2 * 1.2)
ax.set_xlabel('first species')
ax.set_ylabel('second species')
ax.set_title(f'Final state ({first[-1]:.1f}, {second[-1]:.1f})')
ax.legend()
plt.show()
