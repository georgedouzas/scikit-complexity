"""
# Adaptive flow network

The vessels of a grid fed at one corner and drained everywhere else widen when the pressure drop along them is large and
narrow otherwise. The conductances settle where every vessel carries the same pressure drop, so trunks form near the
source and the distal vessels stay thin.
"""

# %%
# The system is built from its class on a grid of 6 by 6 nodes, and the observer saves every frame with the default
# blueprint, whose primary view is the network with the conductances on its edges.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import AdaptiveFlowNetworkSystem

SYSTEM = 'adaptive_flow_network'

system = AdaptiveFlowNetworkSystem().prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, dt=0.1, observers=[observer], until=20)

# %%
# Balanced pressure drops
# -----------------------
# At the fixed point of `dC/dt = C (alpha1 |dp|^gamma - alpha2)` every pressure drop is `(alpha2 / alpha1)^(1 /
# gamma) = 1`, and the conductances grow towards the source at the corner.

conductances = trajectory.get_values('vessels.C')
adaptation = system.interactions[0]
drops = np.abs(adaptation._pressure_drops(conductances[-1]))
graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6), label_attribute='position')
positions = nx.get_node_attributes(graph, 'position')
fig, (network, balance) = plt.subplots(1, 2, figsize=(11, 4.5))
nx.draw_networkx_edges(
    graph,
    positions,
    edgelist=list(adaptation.edges),
    width=1 + 4 * conductances[-1] / conductances[-1].max(),
    ax=network,
)
nx.draw_networkx_nodes(graph, positions, node_size=20, ax=network)
network.set_title('Conductances at the end, source at the corner')
balance.plot(trajectory.times, conductances, lw=0.6)
balance.set_xlabel('time')
balance.set_ylabel('conductance')
balance.set_title(f'Pressure drops at the end: {drops.min():.3f} to {drops.max():.3f}')
fig.tight_layout()
plt.show()
