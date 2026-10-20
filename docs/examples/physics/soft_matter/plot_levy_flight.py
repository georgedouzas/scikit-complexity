"""
# Lévy flight

The walkers jump at every step in a random direction by lengths with no variance. Most jumps are short, but rare jumps
are huge, so the cloud has a dense core and walkers scattered far away by single long jumps.
"""

# %%
# The model is created by name from the catalog with 500 walkers in the plane, whose jump lengths have the tail
# `l^(-2.2)`, and iterated for 200 steps. Every step draws the jump lengths from SciPy's Pareto distribution and the
# directions from normal draws. The observer saves every frame with the default blueprint, whose primary view shows
# the walkers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'levy_flight'

system = create(MODEL, n_walkers=500, mu=1.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# Superdiffusive spreading
# ------------------------
# A sum of `n` jumps with the tail `l^(-1 - mu)` and `mu < 2` grows as `n^(1 / mu)`, by the generalized central limit
# theorem. The median distance from the origin should follow the dashed line of slope `1 / mu = 0.83`, above the slope
# 1/2 of a random walk. The distribution of the jump lengths should follow the power law `l^(-1 - mu)`.

trajectory = simulation.trajectory_
steps = trajectory.times[1:]
positions = trajectory['walkers.x']
distance = np.median(np.linalg.norm(positions[1:], axis=-1), axis=1)
slope = np.polyfit(np.log(steps), np.log(distance), 1)[0]
lengths = np.linalg.norm(np.diff(positions, axis=0), axis=-1).ravel()
edges = np.logspace(0, np.log10(lengths.max()), 30)
density, _ = np.histogram(lengths, bins=edges, density=True)
centers = np.sqrt(edges[1:] * edges[:-1])
fig, (spreading, tail) = plt.subplots(1, 2, figsize=(10, 4))
spreading.loglog(steps, distance, label=f'simulation, slope {slope:.2f}')
spreading.loglog(steps, distance[-1] * (steps / steps[-1]) ** (1 / 1.2), '--', color='black', label='slope 1 / mu')
spreading.loglog(steps, distance[-1] * (steps / steps[-1]) ** 0.5, ':', color='gray', label='random walk, slope 1/2')
spreading.set_xlabel('step n')
spreading.set_ylabel('median distance from the origin')
spreading.set_title('Superdiffusion, |x| ~ n^(1 / mu)')
spreading.legend()
tail.loglog(centers[density > 0], density[density > 0], 'o', label='jump lengths')
tail.loglog(centers, 1.2 * centers ** (-2.2), '--', color='black', label='mu l^(-1 - mu)')
tail.set_xlabel('jump length l')
tail.set_ylabel('probability density')
tail.set_title('Power-law jump lengths')
tail.legend()
fig.tight_layout()
plt.show()
