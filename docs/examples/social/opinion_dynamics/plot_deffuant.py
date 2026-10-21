"""
# Deffuant bounded confidence

Pairs of people meet at random and move their opinions towards each other, but only when the opinions already lie
within a confidence bound. Opinions that start spread uniformly settle into a few separate clusters, about
`1 / (2 d)` of them for the bound `d`, which no longer talk to each other.
"""

# %%
# Two hundred people start with uniform opinions and a confidence bound of 0.2. The model's blueprint shows the
# histogram of the opinions, which collapses into two or three peaks.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'deffuant'

system = create(MODEL, confidence=0.2, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=50)

# %%
# Number of clusters against the bound
# ------------------------------------
# Opinions within 0.01 of each other at the end belong to one cluster. Clusters of fewer than five people are left out,
# since a few stragglers stay between the large clusters. The number of large clusters should follow `1 / (2 d)`,
# rounded down, as the bound shrinks.


def clusters(confidence, seed):
    """Return the number of clusters of at least five people at the end of a run."""
    run = Simulation(create(MODEL, confidence=confidence, random_state=seed), random_state=seed).run(until=50)
    opinions = np.sort(run.trajectory_.agents('people', len(run.trajectory_) - 1).values['opinion'])
    groups = np.split(opinions, np.flatnonzero(np.diff(opinions) > 0.01) + 1)
    return sum(len(group) >= 5 for group in groups)


bounds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
counts = np.array([[clusters(bound, seed) for seed in range(3)] for bound in bounds])
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bounds, counts.mean(axis=1), yerr=counts.std(axis=1), marker='o', capsize=3, label='simulated')
grid = np.linspace(0.08, 0.42, 200)
ax.plot(grid, np.floor(1 / (2 * grid)), color='black', linestyle='--', label='floor(1 / (2 d))')
ax.set_xlabel('confidence bound d')
ax.set_ylabel('clusters of at least five people')
ax.set_title('Narrow bounds split opinions into more clusters')
ax.legend()
plt.show()
