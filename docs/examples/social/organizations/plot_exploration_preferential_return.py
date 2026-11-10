"""
# Exploration and preferential return

Song and coauthors model human mobility by exploration and preferential return, the baseline of the container model. At
every displacement an individual explores a new place with a probability that decays with the number of places it knows,
at a distance from a power law, and otherwise returns to a known place in proportion to its past visits.
"""

# %%
# Ten individuals make 500 displacements each with `rho = 0.6` and `gamma = 0.21`. The default blueprint shows their
# positions and the number of distinct places they know.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import ExplorationPreferentialReturnSystem

SYSTEM = 'exploration_preferential_return'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(ExplorationPreferentialReturnSystem(), observers=[observer], random_state=0)

# %%
# Sublinear exploration
# ---------------------
# The number of distinct places should grow as `n^(1 / (1 + gamma))`, a slope of 0.83 on logarithmic axes, and a
# trajectory should return again and again to a few places.

steps = np.arange(1, len(trajectory.times))
distinct = [np.mean(trajectory.get_agents('individuals', frame).values['distinct']) for frame in steps]
x = [trajectory.get_agents('individuals', frame).values['x'][0] for frame in steps]
y = [trajectory.get_agents('individuals', frame).values['y'][0] for frame in steps]
fig, (growth, path) = plt.subplots(1, 2, figsize=(10, 4))
growth.loglog(steps, distinct, label='simulation')
growth.loglog(steps, distinct[-1] * (steps / steps[-1]) ** (1 / 1.21), ':', color='black', label='slope 1 / 1.21')
growth.set_xlabel('displacements')
growth.set_ylabel('distinct places')
growth.set_title('Exploration slows down')
growth.legend()
path.plot(x, y, '.-', linewidth=0.3, markersize=3)
path.set_xlabel('x')
path.set_ylabel('y')
path.set_title('One trajectory')
fig.tight_layout()
plt.show()
