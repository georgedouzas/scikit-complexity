"""
# Motter-Lai overload cascades

Motter and Lai's cascades of overload failures: the load of a node is the number of shortest paths through it, and
its capacity is a little more than its initial load. Removing the node of the largest load sends its paths through
other nodes, which overload and fail in turn, so one removal can bring down most of a scale-free network, while the
failure of a random node does little harm.
"""

# %%
# The network is a Barabási-Albert graph of 300 nodes with the tolerance `alpha = 0.2`. At every step the laws read
# the betweenness of the nodes on the network present, which NetworkX computes again whenever the network changes, and
# remove every node whose load exceeds its capacity.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import MotterLaiOverloadSystem

SYSTEM = 'motter_lai_overload'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(MotterLaiOverloadSystem(tolerance=0.2), observers=[observer])


def giant_share(tolerance, attack, seed):
    """Return the share of the nodes in the largest component once the cascade stops."""
    system = MotterLaiOverloadSystem(tolerance=tolerance, attack=attack, random_state=seed)
    graph = simulate(system).get_network(-1).get_graph()
    return max(len(component) for component in nx.connected_components(graph)) / 300


# %%
# Largest component against the tolerance
# ---------------------------------------
# After an attack on the node of the largest load the largest component should be a small share of the network at
# low tolerances and recover as the tolerance grows, while a random failure should leave almost all of it, as Motter
# and Lai found for heterogeneous networks.

tolerances = np.linspace(0.0, 1.0, 6)
fig, axis = plt.subplots(figsize=(6, 4))
for attack in ('load', 'degree', 'random'):
    shares = [np.mean([giant_share(tolerance, attack, seed) for seed in range(3)]) for tolerance in tolerances]
    axis.plot(tolerances, shares, 'o-', label=attack)
axis.set_xlabel('tolerance alpha')
axis.set_ylabel('share of nodes in the largest component')
axis.set_title('Cascades after removing one node')
axis.legend()
fig.tight_layout()
plt.show()
