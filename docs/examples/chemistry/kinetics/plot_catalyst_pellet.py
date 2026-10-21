"""
# Catalyst pellet

The surface of a catalyst pellet has two stable states: a reacting state, where the reaction keeps the surface
concentration low, and an inhibited state, where the adsorbed reactant blocks the reaction. Started at the bulk
concentration, the surface settles in the inhibited state.
"""

# %%
# The model is created by name from the catalog with the mass transfer `h = 0.1`, the rate `k = 10`, and the adsorption
# constant `K = 20`, starting at the bulk concentration `c = 1`. The default blueprint shows the surface concentration
# in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'catalyst_pellet'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Bistability
# -----------
# The steady states are the roots of `h (1 - c) (1 + K c)^2 - k c = 0`. The outer two are stable and the middle one
# unstable, so runs started on either side of the middle root should end in different states.

h, k, K = 0.1, 10.0, 20.0
roots = np.sort(np.roots(np.polysub(np.polymul([-h, h], np.polymul([K, 1.0], [K, 1.0])), [k, 0.0])).real)
grid = np.linspace(0.0, 1.0, 400)
fig, (rate, runs) = plt.subplots(1, 2, figsize=(10, 4))
rate.plot(grid, h * (1 - grid) - k * grid / (1 + K * grid) ** 2)
rate.axhline(0.0, color='gray', linewidth=0.8)
rate.plot(roots[[0, 2]], [0, 0], 'o', color='black', label='stable')
rate.plot(roots[1], 0, 'o', color='white', markeredgecolor='black', label='unstable')
rate.set_xlabel('surface concentration c')
rate.set_ylabel('dc/dt')
rate.set_title('Three steady states')
rate.legend()
for c0 in np.linspace(0.02, 1.0, 21):
    run = Simulation(create(MODEL, c=c0), dt=0.5).run(until=100).trajectory_
    runs.plot(run.times, run['pellet.c'], color='C1' if c0 > roots[1] else 'C0', linewidth=0.8)
runs.plot(simulation.trajectory_.times, simulation.trajectory_['pellet.c'], color='black', linewidth=2)
for root in roots:
    runs.axhline(root, color='gray', linestyle='--', linewidth=0.8)
runs.set_xlabel('time')
runs.set_ylabel('surface concentration c')
runs.set_title('Runs split at the unstable state')
fig.tight_layout()
plt.show()
