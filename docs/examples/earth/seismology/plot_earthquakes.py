"""
# Olami-Feder-Christensen earthquakes

The fault is loaded slowly and every site that reaches the threshold slips, passing a fraction of its force to its
neighbours. From random forces the fault organizes into patches of similar force, and earthquakes of all sizes follow
a power law, like the Gutenberg-Richter law of real faults.
"""

# %%
# The model is created by name from the catalog on a 50 by 50 grid with open edges and `alpha = 0.2`, so that every
# slip loses a fifth of its force, and runs for 50,000 earthquakes. The default blueprint shows the forces on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import ccdf, fit_power_law
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'earthquakes'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=250, observers=[observer]).run(until=50000)

# %%
# Gutenberg-Richter law
# ---------------------
# After a transient in which the fault organizes, the sizes of the earthquakes should follow a power law over two
# decades, so their complementary cumulative distribution should follow a straight line on logarithmic axes up to
# sizes of a few hundred slips, and fall off beyond them, where earthquakes reach the size of the grid.

sizes = np.array([event.size for event in simulation.trajectory_.events if event.time > 20000], dtype=float)
fit = fit_power_law(sizes, xmax=300, discrete=True, bootstrap=0, random_state=0)
values, survival = ccdf(sizes)
tail = (values >= fit.xmin) & (values <= 300)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(values, survival, '.', markersize=3, label=f'{len(sizes)} earthquakes')
ax.loglog(
    values[tail],
    survival[tail][0] * (values[tail] / fit.xmin) ** (1 - fit.alpha),
    color='black',
    label=f'power law, exponent {fit.alpha:.2f}',
)
ax.set_xlabel('earthquake size (slips)')
ax.set_ylabel('P(size >= s)')
ax.set_title('Earthquake sizes follow a power law')
ax.legend()
plt.show()
