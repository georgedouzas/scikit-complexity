"""
# Bak-Tang-Wiesenfeld sandpile

Grains drop one at a time on random sites, and a site holding four grains topples, sending one grain to each
neighbour, which may topple in turn; grains that fall off the open edges are lost. Without any tuning the pile
organizes itself into a critical state whose avalanches come in all sizes, from a single toppling to sweeps across
the whole lattice. The view shows the heights of a 48 x 48 pile, from zero (dark) to three grains (bright).
"""

# %%
# The model is created by name from the catalog from random stable heights. One unit of time is one dropped grain
# followed by the full avalanche it triggers. The observer saves every 50th grain with the default blueprint, whose
# primary view is the image of the heights.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import avalanches, histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sandpile'

system = create(MODEL, shape=(48, 48), random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=50, observers=[observer], random_state=0).run(until=14950)

# %%
# Self-organized criticality
# --------------------------
# The mean height should rise from its random start to a plateau near `2.1` grains and stay there, the critical
# state. Each avalanche is recorded as an event with its number of topplings. In the critical state the density of
# sizes should follow a power law `P(s) ~ s^-tau` with `tau` near `1.2` on the square lattice over two decades, up
# to a cutoff set by the size of the lattice where it falls off steeply.

heights = simulation.trajectory_['pile.height'].mean(axis=1)
found = avalanches([record for record in simulation.trajectory_.events if record.time > 5000 and record.size > 0])
edges, density = histogram(found.sizes, 20, log=True)
centers = np.sqrt(edges[1:] * edges[:-1])
shown = density > 0

fig, (pile, laws) = plt.subplots(1, 2, figsize=(10, 4))
pile.plot(np.arange(len(heights)) * 50, heights)
pile.set_xlabel('grains dropped')
pile.set_ylabel('mean height')
pile.set_title('The pile drives itself to the critical state')
laws.loglog(centers[shown], density[shown], 'o', label=f'{len(found.sizes)} avalanches')
guide = np.logspace(0, 2.5, 50)
reference = density[shown][0] * (centers[shown][0] / guide) ** 1.2
laws.loglog(guide, reference, color='black', label='s^-1.2')
laws.set_xlabel('avalanche size s (topplings)')
laws.set_ylabel('probability density')
laws.set_title('Avalanches of all sizes')
laws.legend()
fig.tight_layout()
plt.show()
