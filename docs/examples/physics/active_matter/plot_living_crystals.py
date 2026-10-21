"""
# Living crystals

Light-activated colloids swim and attract each other through the chemical gradients they create, as in the
experiments of Palacci and coworkers. Swimmers that meet stick together, and the clusters merge into crystallites
with hexagonal order that drift and turn as the swimming of their members pulls them around. The view shows 100
colloids of radius 1 in a periodic square of side 50, gathered into a few hexagonal crystallites.
"""

# %%
# The model is created by name from the catalog with swimming speed 3, rotational diffusion 1, and phoretic attraction
# of strength 20 within a range of 10. The catalog registers the Euler-Maruyama stepper with steps of 0.002 for the
# stiff repulsion between colloids, so it is passed explicitly. The observer saves every frame of 0.2 with the default
# blueprint, whose primary view shows the colloids.

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'living_crystals'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.002)
simulation = Simulation(system, solver=solver, dt=0.2, observers=[observer], random_state=0).run(until=40)

# %%
# Clustering by phoretic attraction
# ---------------------------------
# Two colloids belong to one cluster when their centres are closer than 2.5, a little more than a diameter. Without
# attraction the swimmers only meet in brief collisions, so the share of colloids in clusters of three or more should
# stay low and fluctuate; with the attraction it should rise within a few units of time to nearly all colloids and
# stay there, as the crystallites hold together at this strength.


def clustered_share(positions, side=50.0):
    """Return the share of colloids in clusters of at least three at each frame."""
    shares = []
    for frame in positions:
        tree = cKDTree(np.mod(frame, side), boxsize=side)
        _, labels = connected_components(tree.sparse_distance_matrix(tree, 2.5), directed=False)
        sizes = np.bincount(labels)
        shares.append((sizes[labels] >= 3).mean())
    return np.array(shares)


fig, ax = plt.subplots(figsize=(7, 4))
times = simulation.trajectory_.times
ax.plot(times, clustered_share(simulation.trajectory_['swimmers.x']), label='with attraction, v0 = 20')
passive = Simulation(create(MODEL, v0=0.0, random_state=0), solver=solver, dt=0.2, random_state=0).run(until=40)
ax.plot(times, clustered_share(passive.trajectory_['swimmers.x']), label='without attraction, v0 = 0')
ax.set_xlabel('time t')
ax.set_ylabel('share of colloids in clusters of 3 or more')
ax.set_title('Phoretic attraction gathers the swimmers into living clusters')
ax.legend()
fig.tight_layout()
plt.show()
