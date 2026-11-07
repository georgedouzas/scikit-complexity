"""
# Node-merging network

A network of fixed size where a random node merges with one of its neighbours into one node that keeps the links of
both, and a newcomer joins a random node. A node that absorbs a neighbour gains its links, so well-linked nodes grow
faster, and hubs appear without any growth of the network.
"""

# %%
# Five hundred nodes start as a random tree, and every step merges a node with a neighbour and adds a newcomer with
# one link, which keeps the network a tree of the same size.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'node_merging'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), dt=100.0, observers=[observer], random_state=0).prepare()
simulation.run(until=10000)

# %%
# Degree distribution
# -------------------
# The degrees after 20 steps per node should have a fat tail, with hubs far above the largest degree of a random
# graph of the same size and number of links, whose degrees are Poisson.

network = simulation.trajectory_.network(-1)
degrees = np.bincount(np.concatenate([network.sources, network.targets]))[network.nodes]
random = nx.gnm_random_graph(len(network.nodes), len(network.sources), seed=0)
random_degrees = np.array([degree for _, degree in random.degree()])
fig, axis = plt.subplots(figsize=(6, 4))
for values, label in ((degrees, 'merging'), (random_degrees, 'random graph')):
    counts = np.bincount(values)
    support = np.flatnonzero(counts)
    axis.loglog(support, counts[support] / values.size, 'o', label=label)
axis.set_xlabel('degree k')
axis.set_ylabel('share of nodes')
axis.set_title('Merging linked nodes builds hubs')
axis.legend()
fig.tight_layout()
plt.show()
