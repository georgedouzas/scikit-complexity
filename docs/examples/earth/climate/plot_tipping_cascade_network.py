"""
# Network of interacting tipping elements

Forty tipping elements, such as ice sheets and forests, sit in their lower states on a small-world network. The first
element's control is pushed past its critical value, so it tips, and its coupling pushes its neighbours over in turn:
the cascade spreads one link after another, although none of the other elements would tip alone. The view shows the
network midway through the cascade, with the tipped elements bright around the first one and the elements still in
their lower states dark at the far end.
"""

# %%
# The model is created by name from the catalog on a Watts-Strogatz ring of 40 elements, each linked to its four nearest
# neighbours, with a few shortcuts. Every control is 0 except the first, at 1.2, past the critical value 1. The coupling
# `d = 0.2` raises the effective control of an element by `2 d / sqrt(4/27) = 1.04` for each tipped neighbour. The run
# stops at time 18, midway through the cascade, and the observer saves every step with the default blueprint, whose
# primary view is the network colored by the states.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tipping_cascade_network'

graph = nx.connected_watts_strogatz_graph(40, 4, 0.05, seed=3)
controls = np.zeros(40)
controls[0] = 1.2
system = create(MODEL, graph=graph, c=controls, d=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=18)

# %%
# Tipping order and the role of the coupling
# ------------------------------------------
# Run to the end, the cascade travels along the links, so the time an element tips should grow with its graph distance
# from the first element. Swept through the control of the first element, the uncoupled network should tip only that
# element once its control passes 1, while the coupled network should tip all forty, already at a control a little
# below 1, since the neighbours that the first element lifts slightly out of their lower states lift it in return.

trajectory = Simulation(create(MODEL, graph=graph, c=controls, d=0.2), dt=0.1).run(until=40).trajectory_
states = trajectory['elements.x']
tipped_at = trajectory.times[np.argmax(states > 0, axis=0)]
distance = nx.single_source_shortest_path_length(graph, 0)
hops = np.array([distance[node] for node in graph.nodes])

sweep = np.linspace(0.8, 1.2, 9)
counts = {}
for d in (0.0, 0.2):
    counts[d] = []
    for first in sweep:
        controls[0] = first
        final = Simulation(create(MODEL, graph=graph, c=controls, d=d), dt=1.0).run(until=200).trajectory_
        counts[d].append(int((final['elements.x'][-1] > 0).sum()))

fig, (order, cascade) = plt.subplots(1, 2, figsize=(10, 4))
order.plot(hops, tipped_at, 'o')
order.set_xlabel('graph distance from the first element')
order.set_ylabel('time of tipping')
order.set_title('The cascade tips one link after another')
cascade.plot(sweep, counts[0.0], 's-', label='uncoupled, d = 0')
cascade.plot(sweep, counts[0.2], 'o-', label='coupled, d = 0.2')
cascade.axvline(1.0, color='gray', linewidth=0.8, linestyle='--')
cascade.set_xlabel('control c of the first element')
cascade.set_ylabel('tipped elements')
cascade.set_title('Coupling turns one tipping into a cascade')
cascade.legend()
fig.tight_layout()
plt.show()
