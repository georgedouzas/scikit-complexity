"""
# Weakening Ornstein-Uhlenbeck process

Brownian particles are pulled back to the origin by a force that is linear near it but fades as `1 / x` far away. Most
of the time they jitter near the origin like an Ornstein-Uhlenbeck process, but now and then one wanders far out on a
long excursion, since the stationary density has power-law tails instead of Gaussian ones.
"""

# %%
# Four particles start at the origin with `k = 1`, `ell = 1`, and `D = 0.25`, and run for 200 time units. The
# default blueprint shows their positions as time series.

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'weakening_ornstein_uhlenbeck'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=16)
simulation = Simulation(create(MODEL, n_particles=4), dt=0.05, observers=[observer], random_state=0).run(until=200)

# %%
# Power-law tails
# ---------------
# The stationary density is `p(x) ∝ (1 + x²/ℓ²)^(-k ℓ² / 2D)`, here `(1 + x²)^-2`, whose tails fall as `|x|^-4`. The
# positions of 500 particles pooled over the second half of a run of 200 time units should follow it out to
# `|x|` near 5, beyond which the run is too short to fill the slowly relaxing tail, far above the Gaussian of an
# Ornstein-Uhlenbeck process with the same stiffness near the origin, of variance `D / k = 0.25`.

ensemble = Simulation(create(MODEL, n_particles=500), dt=0.05, random_state=1).run(until=200).trajectory_
positions = np.abs(ensemble['particles.x'][2000:].ravel())
edges = np.geomspace(0.05, 30, 40)
density, _ = np.histogram(positions, bins=edges, density=True)
centers = np.sqrt(edges[1:] * edges[:-1])
norm = integrate.quad(lambda x: (1 + x**2) ** -2, 0, np.inf)[0]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(centers, density, 'o', label='simulated, |x|')
ax.loglog(centers, (1 + centers**2) ** -2 / norm, 'k', label=r'$(1 + x^2)^{-2}$')
ax.loglog(centers, 2 * stats.norm.pdf(centers, 0, 0.5), 'k:', label='Ornstein-Uhlenbeck Gaussian')
ax.set_ylim(1e-6, 2)
ax.set_xlabel('distance from the origin |x|')
ax.set_ylabel('density')
ax.set_title('Stationary density with power-law tails')
ax.legend()
plt.show()
