"""
# Jain-Krishna autocatalytic networks

Molecular species catalyse each other, and the least populated species is replaced at every graph update by a new one
with random links. For a long time the network stays sparse and few species hold the population. Once a cycle of
catalysts forms by chance it wins the selection, and the autocatalytic set grows until it spans nearly every species.
"""

# %%
# The model is created by name from the catalog with 40 species, links of probability 0.02, and a graph update every
# 10 steps, saved every fifth step in 299 frames. The observer uses the default blueprint, whose primary view is the
# catalytic network colored by the population of every species.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'jain_krishna'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=5.0, observers=[observer], random_state=1).run(until=1490)

# %%
# The rise of the autocatalytic set
# ---------------------------------
# The Perron eigenvalue of the catalytic network is 0 without cycles and at least 1 with an autocatalytic set. It
# should jump to 1 when the populated species rise from a few to nearly all, and the number of links should grow
# afterwards.

trajectory = simulation.trajectory_
populated = np.nansum(trajectory['species.x'] > 1e-3, axis=1)
eigenvalues, links = [], []
for frame in range(len(trajectory)):
    snapshot = trajectory.network(frame)
    graph = nx.DiGraph()
    graph.add_nodes_from(snapshot.nodes.tolist())
    graph.add_edges_from(zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True))
    adjacency = nx.to_numpy_array(graph, nodelist=snapshot.nodes.tolist())
    eigenvalues.append(max(abs(np.linalg.eigvals(adjacency))))
    links.append(snapshot.sources.size)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trajectory.times, populated, label='populated species')
ax.plot(trajectory.times, links, label='links')
ax.plot(trajectory.times, 10 * np.asarray(eigenvalues), label='10 x Perron eigenvalue')
ax.set_xlabel('step')
ax.set_title('An autocatalytic set forms and spans the network')
ax.legend()
plt.show()
