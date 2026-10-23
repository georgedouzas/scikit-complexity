"""
# Quenched mean-field SIS

Every node of a scale-free network carries its own probability of being infected. Above the epidemic threshold, the
inverse of the largest eigenvalue of the adjacency matrix, the infection settles at an endemic state where the hubs are
almost surely infected and the poorly connected nodes rarely are.
"""

# %%
# The model is created by name from the catalog on its default Barabási-Albert network of 300 nodes, with a spreading
# rate twice the threshold. The observer saves every frame with the default blueprint, whose primary view is the
# network colored by the infection probability of every node.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'quenched_mean_field_sis'

graph = create(MODEL).elements[0].space.graph
threshold = 1 / max(np.linalg.eigvalsh(nx.to_numpy_array(graph)))
system = create(MODEL, lam=2 * threshold)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=40)

# %%
# Infection probability by degree
# -------------------------------
# At the endemic state `p_i = lam s_i / (1 + lam s_i)` with `s_i = sum_j A_ij p_j`, so the probability rises with the
# degree. Every node should fall on that relation, and a run below the threshold should die out.

final = simulation.trajectory_['nodes.p'][-1]
adjacency = nx.to_numpy_array(graph)
pressure = 2 * threshold * adjacency @ final
below = Simulation(create(MODEL, lam=0.5 * threshold), dt=0.5).run(until=40).trajectory_['nodes.p'][-1]
degrees = adjacency.sum(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(degrees, final, 'o', alpha=0.6, label='simulated')
order = np.argsort(degrees)
ax.plot(degrees[order], (pressure / (1 + pressure))[order], '--', color='black', label='stationary relation')
ax.set_xscale('log')
ax.set_xlabel('degree')
ax.set_ylabel('infection probability p_i')
ax.set_title(f'Endemic above 1/Lambda = {threshold:.3f}; below it max p = {below.max():.1e}')
ax.legend()
plt.show()
