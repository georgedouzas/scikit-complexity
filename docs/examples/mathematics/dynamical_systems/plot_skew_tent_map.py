"""
# Skew tent map

One hundred points start in an interval of width 0.01 and spread over the whole unit interval within about ten steps.
After that they cover it evenly, since the skew tent map keeps the uniform density for any position of its top.
"""

# %%
# The ensemble iterates the map with its top at `a = 0.3` for 40 steps. The default blueprint shows the values of all
# points as time series, which fan out from the starting interval and then fill the unit interval.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'skew_tent_map'

system = create(MODEL, n_points=100, a=0.3, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=40)

# %%
# Return map and invariant density
# --------------------------------
# Successive values of every point should lie on the two branches of the tent, `x / a` and `(1 - x) / (1 - a)`. The
# values of the last 30 steps, pooled over the ensemble, should give a flat histogram at the density 1 that Bertin
# derives from the Perron-Frobenius equation `p(x) = a p(ax) + (1 - a) p(1 - (1 - a) x)`.

values = simulation.trajectory_['points.x']
fig, (branches, density) = plt.subplots(1, 2, figsize=(10, 4))
branches.plot(values[:-1].ravel(), values[1:].ravel(), '.', markersize=1, alpha=0.3)
branches.set_xlabel('$x_t$')
branches.set_ylabel('$x_{t+1}$')
branches.set_title('Return map: the two branches of the tent')
density.hist(values[10:].ravel(), bins=20, range=(0, 1), density=True, label='simulated, last 30 steps')
density.axhline(1.0, color='black', label='invariant density $p(x) = 1$')
density.set_ylim(0, 1.5)
density.set_xlabel('x')
density.set_ylabel('density')
density.set_title('The ensemble is uniform on [0, 1]')
density.legend(loc='lower center')
fig.tight_layout()
plt.show()
