"""
# Self-organized branching process

Avalanches spread down a binary tree: every active site wakes two sites below it with the probability `p`. Energy enters
with every avalanche and leaves through the sites that reach the bottom, and this balance tunes `p` to the critical
value `1 / 2` without any adjustment from outside, a mean-field picture of self-organized criticality.
"""

# %%
# The tree has ten generations, and the branching probability starts at 0.3. One step is one generation of an
# avalanche. The default blueprint shows `p`, the active sites, and the generation.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.criticality import SelfOrganizedBranchingSystem

SYSTEM = 'self_organized_branching'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=50)
trajectory = SelfOrganizedBranchingSystem().prepare().simulate(until=30000, observers=[observer], random_state=0)

# %%
# Approach to the critical point
# ------------------------------
# Started below or above `1 / 2`, the branching probability should drift to it and then fluctuate around it.

fig, ax = plt.subplots(figsize=(6, 4))
for start in (0.3, 0.7):
    run = SelfOrganizedBranchingSystem(p=start).prepare().simulate(until=30000, random_state=0)
    ax.plot(np.asarray(run.times), np.ravel(run.get_values('tree.p')), label=f'p(0) = {start}')
ax.axhline(0.5, linestyle='--', color='black')
ax.set_xlabel('generations')
ax.set_ylabel('branching probability p')
ax.set_title('Self-organization to p = 1/2')
ax.legend()
plt.show()

# %%
# Sizes of avalanches
# -------------------
# Every avalanche is an `avalanche` event of the run, whose profile holds the active sites of every generation and
# whose size is their sum. Once `p` is near `1 / 2` the process is a critical branching process, so the sizes should
# follow the mean-field power law `S^(-3/2)` up to the cutoff that the ten generations set.

sizes = np.asarray([event.size for event in trajectory.events if event.time > 10000])
histogram = distribution(sizes, kind='histogram', bins=15, log=True)
centers = np.sqrt(histogram.edges[1:] * histogram.edges[:-1])
shown = histogram.density > 0

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(centers[shown], histogram.density[shown], 'o', label='simulation')
ax.loglog(
    centers[shown], histogram.density[shown][0] * (centers[shown] / centers[shown][0]) ** -1.5, '--', label='S^(-3/2)'
)
ax.set_xlabel('size of an avalanche')
ax.set_ylabel('density')
ax.set_title('Critical avalanches of the branching process')
ax.legend()
plt.show()
