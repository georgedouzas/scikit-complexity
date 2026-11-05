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

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hypergraph_random_walk'

system = create(MODEL, seed=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=40)

# %%
# Stationary distribution
# -----------------------
# The probability at every node should approach `D_i / sum_l D_l`, where `D_i` sums `(|e| - 1)^2` over the hyperedges
# `e` of node `i`.

probabilities = np.asarray(simulation.trajectory_.values('nodes.p'))[-1]
weights = np.asarray(system.elements[0].values['D'])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(weights / weights.sum(), probabilities, 'o')
ax.plot([0, probabilities.max()], [0, probabilities.max()], '--', color='black')
ax.set_xlabel('weight of the hyperedges of the node, normalized')
ax.set_ylabel('probability of the walker at t = 40')
ax.set_title('Size-biased random walk on a hypergraph')
plt.show()
