"""
# Site percolation

Each site of a square lattice is occupied with probability `p`, and a fire lit along the top row burns, one layer per
step, every cluster of occupied sites connected to it. Above the threshold `p_c = 0.5927` a cluster spans the lattice
and the fire reaches the bottom row; just above it, as here, the burnt cluster is a sparse fractal full of holes and
dead ends. The view shows a 128 x 128 lattice at `p = 0.6` once the fire has gone out, with burnt sites the brightest.
"""

# %%
# The model is created by name from the catalog. One step moves the fire front by one site. The observer saves every
# step with the default blueprint, whose primary view is the image of the covers: empty, occupied, burning, and burnt.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'site_percolation'

system = create(MODEL, shape=(128, 128), p=0.6, random_state=2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=299)

# %%
# Spanning probability and finite-size scaling
# --------------------------------------------
# The fire spans the lattice when it reaches the bottom row. On an infinite lattice the spanning probability jumps
# from zero to one at `p_c`; on finite lattices it rises over a window that shrinks as `L^(-1 / nu)` with
# `nu = 4 / 3`. Over 30 lattices per point, the curves of larger lattices should be steeper and all of them should
# cross near `p_c`.


def spans(side, p, seed):
    """Return whether the fire lit along the top row of a lattice reaches its bottom row."""
    run = Simulation(create(MODEL, shape=(side, side), p=p, random_state=seed), dt=1).run(until=4 * side)
    return bool((run.trajectory_['lattice.cover'][-1].reshape(side, side)[-1] >= 2).any())


probabilities = np.linspace(0.5, 0.7, 17)
fig, ax = plt.subplots(figsize=(7, 4))
for side in (16, 32, 64):
    spanning = [np.mean([spans(side, p, seed) for seed in range(30)]) for p in probabilities]
    ax.plot(probabilities, spanning, 'o-', label=f'L = {side}')
ax.axvline(0.5927, color='gray', linestyle='--', label='p_c = 0.5927')
ax.set_xlabel('occupation probability p')
ax.set_ylabel('spanning probability')
ax.set_title('The spanning transition sharpens with the lattice size')
ax.legend()
fig.tight_layout()
plt.show()
