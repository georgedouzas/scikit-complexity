"""
# Motility-induced phase separation

Self-propelled disks that only repel each other still separate into a dense and a dilute phase. A swimmer that runs
into a crowd is blocked until its heading turns away, which takes the long persistence time `1 / D_r`, and more
swimmers pile up behind it meanwhile. Fast, persistent swimmers therefore collect into dense clusters surrounded by
a dilute gas. The view shows 600 disks at packing fraction 0.5 that have gathered into clusters.
"""

# %%
# The model is created by name from the catalog with 600 disks of unit diameter, packing fraction 0.5, speed `v = 1`,
# and rotational diffusion `D_r = 0.01`, a Péclet number `v / (D_r sigma) = 100`. The Euler-Maruyama stepper takes
# steps of 0.005 and the observer saves every frame of 0.5 with the default blueprint, whose primary view shows the
# disks.

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'motility_induced_phase_separation'

system = create(MODEL, n_particles=600, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.005)
simulation = Simulation(system, solver=solver, dt=0.5, observers=[observer], random_state=0).run(until=120)

# %%
# Bimodal local density
# ---------------------
# The local packing fraction around each disk is the area of the disks within a distance 2.5, the disk itself
# included, over the area of that circle. The disks start uniformly spread, with local packing fractions around the
# mean 0.5; once the phases have separated the distribution should split into a dilute peak and a dense peak near
# close packing.

side = np.sqrt(600 * np.pi / 4 / 0.5)
radius = 2.5


def local_packing(positions):
    """Return the local packing fraction around every disk of a periodic box."""
    tree = cKDTree(np.mod(positions, side), boxsize=side)
    counts = np.array([len(item) for item in tree.query_ball_point(np.mod(positions, side), radius)])
    return counts * (np.pi / 4) / (np.pi * radius**2)


positions = simulation.trajectory_['swimmers.x']
fig, ax = plt.subplots(figsize=(7, 4))
bins = (np.arange(27) + 0.5) * (np.pi / 4) / (np.pi * radius**2)
ax.hist(local_packing(positions[0]), bins=bins, density=True, alpha=0.5, label='start, uniform disks')
late = np.concatenate([local_packing(frame) for frame in positions[-20:]])
ax.hist(late, bins=bins, density=True, alpha=0.5, label='end, separated phases')
ax.axvline(0.5, color='gray', linestyle='--', label='mean packing fraction')
ax.set_xlabel('local packing fraction')
ax.set_ylabel('probability density')
ax.set_title('The local density splits into a gas and a dense phase')
ax.legend()
fig.tight_layout()
plt.show()
