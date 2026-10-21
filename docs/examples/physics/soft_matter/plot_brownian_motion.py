"""
# Brownian motion

Particles in a fluid are kicked by the molecules around them and slowed by friction. Over times shorter than the
momentum relaxation time `m / gamma` they keep their thermal velocity and fly ballistically; over longer times the
kicks randomize their direction and they diffuse. The view shows 500 particles released together at the origin of
the plane, spreading into a round Gaussian cloud.
"""

# %%
# The model is created by name from the catalog with inertia, mass `m = 1`, friction `gamma = 1`, and thermal energy
# `kT = 1`, so the momentum relaxation time is 1. The Euler-Maruyama stepper takes steps of 0.005 and the observer
# saves every frame of 0.05 with the default blueprint, whose primary view shows the particles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'brownian_motion'

system = create(MODEL, n_particles=500, n_dim=2, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.005)
simulation = Simulation(system, solver=solver, dt=0.05, observers=[observer], random_state=0).run(until=10)

# %%
# From ballistic flight to diffusion
# ----------------------------------
# The Ornstein-Uhlenbeck solution of the Langevin equation gives the mean squared displacement per axis
# `<x^2> = 2 kT / gamma (t - tau (1 - exp(-t / tau)))` with `tau = m / gamma`. It grows as `kT t^2 / m` below `tau`
# and as `2 kT t / gamma` above it, and the measured displacement in the plane, twice the value per axis, should
# follow it across the crossover.

trajectory = simulation.trajectory_
times = trajectory.times[1:]
squared = np.mean(np.sum(trajectory['beads.x'][1:] ** 2, axis=-1), axis=1)
exact = 2 * 2 * (times - (1 - np.exp(-times)))
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(times, squared, 'o', markersize=3, label='simulation, 500 particles')
ax.loglog(times, exact, color='black', label='Ornstein-Uhlenbeck')
ax.loglog(times, 2 * times**2, ':', color='gray', label='ballistic, 2 kT t^2 / m')
ax.loglog(times, 4 * times, '--', color='gray', label='diffusive, 4 kT t / gamma')
ax.axvline(1.0, color='lightgray', label='m / gamma')
ax.set_ylim(squared.min() / 2, squared.max() * 2)
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement in the plane')
ax.set_title('Ballistic below m / gamma, diffusive above')
ax.legend()
fig.tight_layout()
plt.show()
