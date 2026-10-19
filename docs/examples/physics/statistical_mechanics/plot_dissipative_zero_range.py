"""
# Dissipative zero-range process

Energy quanta enter the root of a binary tree from a hot reservoir and cascade down to the leaves, where they are
dissipated. The image shows the five levels of the tree on rings around the root, with the leaves on the outer ring
and every node colored by its number of quanta: the root holds the most quanta, and the outer rings are mostly empty.
"""

# %%
# The model is created by name from the catalog with five levels of two branches, transfer rates `nu_j = k_j^0.5`,
# a hot reservoir at `beta_ext = 0.2` that makes the cascade visible, and a dissipation rate 1 at the leaves. The
# observer saves every frame with the default blueprint, whose primary view is the drawing of the tree. The Numba
# backend runs the transfers in a compiled loop.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'dissipative_zero_range'

system = create(MODEL, n_levels=5, branches=2, alpha=0.5, beta_ext=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0, backend='numba').run(until=200)

# %%
# Temperature profile across scales
# ---------------------------------
# The stationary state is a product of geometric laws `exp(-beta_j n epsilon0)` per level, with `z_j = exp(-beta_j
# epsilon0)` solving the balance of the currents `nu_j (z_j - z_(j + 1))` of Bertin's equations (3.56) to (3.58). The
# probability that a node of level `j` holds a quantum is `z_j`, so `beta_j = -ln P(n > 0)`. For `alpha > -1` the
# inverse temperature should grow across the levels, with a finite energy flux, and for `alpha < -1` it should stay
# close to `beta_ext` down to the last levels, where the dissipation of the finite tree sets in. These runs use the
# reservoir at `beta_ext = 1` of Bertin's figure 3.4.


def balance_profile(n_levels, branches, alpha, *, nu=1.0, beta_ext=1.0, dissipation=1.0):
    """Return the inverse temperatures `beta_j` that solve the balance of currents of every level."""
    rates = nu * (branches ** np.arange(n_levels - 1)) ** alpha
    matrix, source = np.zeros((n_levels, n_levels)), np.zeros(n_levels)
    source[0], matrix[0, 0] = nu * np.exp(-beta_ext), nu
    for level in range(n_levels - 1):
        matrix[level, level] += branches * rates[level]
        matrix[level, level + 1] -= branches * rates[level]
        matrix[level + 1, level + 1] += rates[level]
        matrix[level + 1, level] -= rates[level]
    matrix[-1, -1] += dissipation
    return -np.log(np.linalg.solve(matrix, source))


levels = np.arange(1, 6)
starts = 2 ** np.arange(6) - 1
fig, profile = plt.subplots(figsize=(6, 4))
for alpha in (0.5, -2.0):
    run = Simulation(create(MODEL, alpha=alpha), dt=1.0, random_state=0, backend='numba').run(until=4000)
    quanta = run.trajectory_['nodes.n'][100:] > 0
    measured = [-np.log(quanta[:, start:stop].mean()) for start, stop in itertools.pairwise(starts)]
    profile.plot(levels, measured, 'o', label=f'simulation, alpha = {alpha}')
    profile.plot(levels, balance_profile(5, 2, alpha), color='black')
profile.axhline(1.0, color='gray', linestyle='--', label='beta_ext')
profile.set_xlabel('level j')
profile.set_ylabel('inverse temperature beta_j')
profile.set_title('Energy cascade: beta_j against the balance of currents (lines)')
profile.legend()
fig.tight_layout()
plt.show()
