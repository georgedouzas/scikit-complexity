"""
# Random walk on a hypergraph

A walker on a hypergraph crosses a hyperedge to each of its other members with a weight that grows with the size of
the hyperedge. The probability of the walker spreads from one node and settles on the weight of the hyperedges around
every node.
"""

# %%
# The hypergraph holds 30 nodes and random hyperedges of sizes 2 to 5. The walker starts at node 0.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import HypergraphRandomWalkSystem

SYSTEM = 'hypergraph_random_walk'

system = HypergraphRandomWalkSystem(seed=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=40, dt=0.5, observers=[observer])

# %%
# Stationary distribution
# -----------------------
# The probability at every node should approach `D_i / sum_l D_l`, where `D_i` sums `(|e| - 1)^2` over the hyperedges
# `e` of node `i`.

probabilities = np.asarray(trajectory.get_values('nodes.p'))[-1]
weights = np.asarray(system.prepare().system_.elements[0].values['D'])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(weights / weights.sum(), probabilities, 'o')
ax.plot([0, probabilities.max()], [0, probabilities.max()], '--', color='black')
ax.set_xlabel('weight of the hyperedges of the node, normalized')
ax.set_ylabel('probability of the walker at t = 40')
ax.set_title('Size-biased random walk on a hypergraph')
plt.show()
