"""
# Random walker epidemics

One sick walker among a thousand random walkers starts a deadly epidemic. The sick infect the walkers they meet and
die some steps later, so the disease spreads outwards as a front, like a forest fire, and burns out behind it.
"""

# %%
# The model is created by name from the catalog with a thousand walkers on a 64 by 64 grid. The observer saves every
# step with the default blueprint, whose primary view shows the walkers on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'random_walker_epidemics'

system = create(MODEL, walkers=1000, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=299)

# %%
# Density threshold
# -----------------
# Above a critical density of walkers the epidemic should percolate and kill a large share of them, while below it
# the disease should die out after a few victims.


def dead_share(walkers, seed):
    """Return the share of walkers dead at the end of a run."""
    run = Simulation(create(MODEL, walkers=walkers, random_state=seed), dt=1, random_state=seed).run(until=300)
    final = run.trajectory_.agents('walkers', len(run.trajectory_.times) - 1)
    return np.mean(final.values['health'] == 2)


counts = [100, 250, 500, 1000, 2000]
shares = [np.mean([dead_share(count, seed) for seed in range(3)]) for count in counts]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.array(counts) / 64**2, shares, 'o-')
ax.set_xscale('log')
ax.set_xlabel('walkers per cell')
ax.set_ylabel('share of walkers dead after 300 steps')
ax.set_title('Dense populations let the epidemic spread')
plt.show()
