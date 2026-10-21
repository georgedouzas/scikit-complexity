"""
# Particle in a well

A particle in the well `U(x) = k x² e^-x` rocks back and forth while its energy is below the barrier at `x = 2`, and
escapes over the barrier to infinity once its energy exceeds it, so the phase plane splits into trapped and free motion.
"""

# %%
# The particle starts at `x = 1` with velocity 0.5, energy about 0.86, below the barrier height `4 k e^-2 ≈ 1.08`
# for `k = 2`. The default blueprint shows the position, an anharmonic oscillation that leans towards the barrier.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'particle_in_a_well'

system = create(MODEL, x=1.0, v=0.5, k=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=25)

# %%
# Trapped and escaping orbits
# ---------------------------
# The energy `E = v^2 / 2 + k x^2 e^-x` is conserved, so each orbit should follow one level curve of `E`. Orbits with
# `E` below the barrier, launched from `x = 1`, should close around the bottom of the well, while those above it should
# cross `x = 2` and run away. The dashed separatrix is the level curve through the barrier top.

grid_x, grid_v = np.meshgrid(np.linspace(-1.0, 6.0, 300), np.linspace(-2.0, 2.0, 300))
energy = grid_v**2 / 2 + 2 * grid_x**2 * np.exp(-grid_x)
fig, ax = plt.subplots(figsize=(6, 4))
ax.contour(grid_x, grid_v, energy, levels=[8 * np.exp(-2)], colors='black', linestyles='--')
for speed in [0.2, 0.5, 0.8, 0.9]:
    run = Simulation(create(MODEL, x=1.0, v=speed, k=2.0), dt=0.02).run(until=25)
    position, velocity = np.ravel(run.trajectory_['particle.x']), np.ravel(run.trajectory_['particle.v'])
    inside = position < 6
    ax.plot(position[inside], velocity[inside], label=f'v0 = {speed}')
ax.set_xlim(-1.0, 6.0)
ax.set_xlabel('x')
ax.set_ylabel('v')
ax.set_title('Phase portrait: trapped below the separatrix, free above')
ax.legend()
plt.show()
