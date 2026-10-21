"""
# Van der Pol ice ages

Ice volume `x` builds up slowly over a long glacial, then the climate state `y` flips and the ice collapses in a fast
termination: a relaxation oscillation with the sawtooth shape of the glacial cycles of the last million years.
"""

# %%
# The model is created by name from the catalog with the time scale separation `alpha = 30`, the asymmetry
# `beta = 0.75`, and no orbital forcing, and runs for 400 kyr. The default blueprint shows `x` and `y` in one time
# series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'van_der_pol_ice_ages'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=400)

# %%
# Slow manifold and fast jumps
# ----------------------------
# With a large `alpha`, `y` stays on the outer branches of the cubic nullcline `x = y^3 / 3 - y` while `x` drifts
# slowly, and jumps across between the branches at its folds `x = -2/3` and `x = 2/3`. The limit cycle should hug the
# outer branches and cross between them almost horizontally.

trajectory = simulation.trajectory_
settled = trajectory.times > 100
grid = np.linspace(-2.3, 2.3, 300)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(grid**3 / 3 - grid, grid, color='gray', linestyle='--', label='nullcline x = y^3 / 3 - y')
ax.plot(trajectory['climate.x'][settled], trajectory['climate.y'][settled], linewidth=2, label='limit cycle')
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel('ice volume x')
ax.set_ylabel('climate state y')
ax.set_title('Slow drift along the branches, fast terminations')
ax.legend()
plt.show()
