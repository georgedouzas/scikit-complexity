"""
# Bak-Tang-Wiesenfeld sandpile

Grains drop one at a time on random sites, and a site holding four grains topples, sending one grain to each
neighbour, which may topple in turn. Grains that fall off the open edges are lost. Without any tuning the pile
organizes itself into a critical state whose avalanches come in all sizes, from a single toppling to sweeps across
the whole lattice. The view shows the heights of a 48 x 48 pile, from zero (dark) to three grains (bright).
"""

# %%
# The example builds `SandpileSystem` from random stable heights. One unit of time is one dropped grain
# followed by the full avalanche it triggers. The observer saves every 50th grain with the default blueprint, whose
# primary view is the image of the heights.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution, events
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.criticality import SandpileSystem, abelian_sandpile

SYSTEM = 'sandpile'

system = SandpileSystem(shape=(48, 48), random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = system.prepare().simulate(until=14950, dt=50, observers=[observer], random_state=0)

# %%
# Self-organized criticality
# --------------------------
# The mean height should rise from its random start to a plateau near `2.1` grains and stay there, the critical
# state. Each avalanche is recorded as an event with its number of topplings. In the critical state the density of
# sizes should follow a power law `P(s) ~ s^-tau` with `tau` near `1.2` on the square lattice over two decades, up
# to a cutoff set by the size of the lattice where it falls off steeply.

heights = trajectory.get_values('pile.height').mean(axis=1)
found = events(
    [record for record in trajectory.events if record.time > 5000 and record.size > 0],
    kind='excursions',
)
histogram = distribution(found.sizes, kind='histogram', bins=20, log=True)
edges, density = histogram.edges, histogram.density
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

# %%
# Dhar's mean avalanche size
# --------------------------
# The sandpile is abelian, so its toppling matrix gives its statistics exactly. A grain that falls on a random site
# makes on average `(1 / N) sum_ij G_ij` topplings, where `G` is the inverse of the toppling matrix, the lattice Green
# function. The topplings per grain of the simulated pile after its transient should agree with that mean size, which
# grows as the squared side of the lattice.

dropped = trajectory.times[-1] - 5000
simulated = sum(record.size for record in trajectory.events if record.time >= 5000) / dropped
sides = np.array([8, 16, 32, 48, 64])
means = [abelian_sandpile(SandpileSystem(shape=(side, side)), kind='mean-size').mean_size for side in sides]

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(sides, means, 'o-', label='mean size from the toppling matrix')
ax.loglog([48], [simulated], 's', color='black', label='simulated 48 x 48 pile')
ax.set_xlabel('side of the lattice')
ax.set_ylabel('topplings per grain')
ax.set_title('The mean avalanche size grows as the squared side')
ax.legend()
fig.tight_layout()
plt.show()
