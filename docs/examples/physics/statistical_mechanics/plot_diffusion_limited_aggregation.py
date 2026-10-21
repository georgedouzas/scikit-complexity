"""
# Diffusion-limited aggregation

Random walkers wander on a grid and stick for good when they touch the aggregate grown from a seed at the centre.
The tips of the aggregate catch the walkers before they can reach the fjords between the branches, so the tips grow
fastest and the aggregate becomes a branched, tenuous fractal. The view shows the walkers on a 101 x 101 grid, the
stuck ones bright, with the aggregate branching out from the centre.
"""

# %%
# The model is created by name from the catalog with 800 walkers, the seed included, scattered over a periodic grid
# of side 101. The observer saves every tenth step with the default blueprint, whose primary view shows the walkers
# colored by whether they are stuck.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'diffusion_limited_aggregation'

system = create(MODEL, n_walkers=800, size=101, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=10, observers=[observer], random_state=0).run(until=2990)

# %%
# Fractal dimension from the mass-radius relation
# -----------------------------------------------
# The number of stuck walkers within a distance `r` of the seed grows as `r^d_f`, with the fractal dimension
# `d_f = 1.71` of two-dimensional diffusion-limited aggregation, well below the dimension 2 of a compact cluster.
# The counts of the final aggregate, fitted over the inner half of its radii, should follow a power law with a slope
# near this value and well below 2, and level off beyond the tips of the aggregate.

final = simulation.trajectory_.agents('walkers', len(simulation.trajectory_) - 1)
stuck = final.locations[final.values['stuck'] > 0].astype(float)
distances = np.linalg.norm(stuck - 50, axis=1)
radii = np.arange(2, int(distances.max()) + 1)
mass = np.array([(distances <= radius).sum() for radius in radii])
fitted = slice(1, len(radii) // 2)
slope = np.polyfit(np.log(radii[fitted]), np.log(mass[fitted]), 1)[0]
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(radii, mass, 'o', markersize=3, label=f'aggregate of {len(stuck)} walkers, slope {slope:.2f}')
ax.loglog(radii, mass[fitted][0] * (radii / radii[fitted][0]) ** 1.71, '--', color='black', label='r^1.71')
ax.loglog(radii, mass[fitted][0] * (radii / radii[fitted][0]) ** 2, ':', color='gray', label='compact, r^2')
ax.set_xlabel('distance r from the seed')
ax.set_ylabel('stuck walkers within r')
ax.set_title('The aggregate is a fractal of dimension near 1.7')
ax.legend()
fig.tight_layout()
plt.show()
