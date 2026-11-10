"""
# Davidsen-Ebel-Bornholdt network

In a network of fixed size, a random node introduces two of its friends to each other, closing a triangle, while nodes
leave at random and newcomers arrive with one friend. Closing triangles keeps the clustering high, and since a node is
the friend of many others in proportion to its degree, it gains links by preferential attachment.
"""

# %%
# Two hundred nodes start as a random graph. Every step a node drawn uniformly closes a triangle, and with probability
# `turnover = 0.1` a node drawn uniformly is replaced by a newcomer with one link.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import DavidsenEbelBornholdtSystem

SYSTEM = 'davidsen_ebel_bornholdt'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(DavidsenEbelBornholdtSystem(), until=10000, dt=100.0, observers=[observer], random_state=0)


def graphs(turnover, until=10000):
    """Return the networks of the steady state, every 200 steps after the first fifth of the run."""
    run = simulate(DavidsenEbelBornholdtSystem(turnover=turnover), until=until, random_state=0)
    found = []
    for frame in range(until // 5, until + 1, 200):
        network = run.get_network(frame)
        graph = nx.Graph(list(zip(network.sources.tolist(), network.targets.tolist(), strict=True)))
        graph.add_nodes_from(network.nodes.tolist())
        found.append(graph)
    return found


# %%
# Mean degree and clustering
# --------------------------
# A closure adds one link per step, and a turnover removes a node of mean degree `k` and adds one link, so the mean
# degree should settle at `1 + 1 / turnover`. The clustering should stay far above the `k / N` of a random graph.

turnovers = np.array([0.05, 0.1, 0.2, 0.4])
degrees, clustering = [], []
for turnover in turnovers:
    steady = graphs(turnover)
    degrees.append(np.mean([2 * graph.number_of_edges() / graph.number_of_nodes() for graph in steady]))
    clustering.append(np.mean([nx.average_clustering(graph) for graph in steady]))
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
left.plot(turnovers, degrees, 'o', label='simulated')
grid = np.linspace(0.04, 0.45, 100)
left.plot(grid, 1 + 1 / grid, 'k:', label='1 + 1 / turnover')
left.set_xlabel('turnover')
left.set_ylabel('mean degree')
left.legend()
right.plot(turnovers, clustering, 'o', label='simulated')
right.plot(turnovers, np.array(degrees) / 200, 'k:', label='random graph, k / N')
right.set_xlabel('turnover')
right.set_ylabel('mean clustering')
right.legend()
fig.suptitle('Turnover sets the degree, closure keeps the clustering')
fig.tight_layout()
plt.show()
