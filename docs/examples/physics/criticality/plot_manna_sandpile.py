"""
# Manna sandpile

Grains drop one at a time on random sites, and a site holding two or more grains topples, sending two grains each to
a randomly chosen neighbour; grains that fall off the open edges are lost. The randomness of the toppling puts the
Manna model in another universality class than the deterministic Bak-Tang-Wiesenfeld pile, but it organizes itself
into a critical state in the same way, with avalanches of all sizes. The view shows the heights of a 32 x 32 pile,
empty (dark) or holding one grain (bright), between avalanches.
"""

# %%
# The model is created by name from the catalog from random stable heights. One unit of time is one dropped grain
# followed by the full avalanche it triggers. The observer saves every 30th grain with the default blueprint, whose
# primary view is the image of the heights.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import avalanches, histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'manna_sandpile'

system = create(MODEL, shape=(32, 32), random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=30, observers=[observer], random_state=0).run(until=8970)

# %%
# Self-organized criticality
# --------------------------
# The density of grains should rise from its random start to a plateau near the critical density `0.68` of the
# two-dimensional Manna class and stay there. Each avalanche is recorded as an event with its number of topplings.
# In the critical state the density of sizes should follow a power law `P(s) ~ s^-tau` with `tau` near `1.27`, the
# Manna exponent, over two decades, up to a cutoff set by the size of the lattice where it falls off steeply.

heights = simulation.trajectory_['pile.height'].mean(axis=1)
found = avalanches([record for record in simulation.trajectory_.events if record.time > 3000 and record.size > 0])
edges, density = histogram(found.sizes, 20, log=True)
centers = np.sqrt(edges[1:] * edges[:-1])
shown = density > 0

fig, (pile, laws) = plt.subplots(1, 2, figsize=(10, 4))
pile.plot(np.arange(len(heights)) * 30, heights)
pile.set_xlabel('grains dropped')
pile.axhline(0.68, color='gray', linestyle='--', label='critical density 0.68')
pile.legend()
pile.set_ylabel('grains per site')
pile.set_title('The pile drives itself to the critical state')
laws.loglog(centers[shown], density[shown], 'o', label=f'{len(found.sizes)} avalanches')
guide = np.logspace(0, 2.5, 50)
reference = density[shown][0] * (centers[shown][0] / guide) ** 1.27
laws.loglog(guide, reference, color='black', label='s^-1.27')
laws.set_xlabel('avalanche size s (topplings)')
laws.set_ylabel('probability density')
laws.set_title('Avalanches of all sizes')
laws.legend()
fig.tight_layout()
plt.show()
