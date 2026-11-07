"""
# Explosive percolation

Edges join isolated nodes one at a time. Every step draws two candidate edges and adds the one whose product of the
sizes of the components it joins is smaller, the product rule of Achlioptas, D'Souza, and Spencer. The rule holds back
the growth of large components, until the giant component appears in a sudden jump.
"""

# %%
# A thousand isolated nodes gain one edge per step. The size of the component of every node is read at the start of
# every step, and the score of a candidate edge is the product of the sizes of its two ends.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'explosive_percolation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=1000)


def largest(candidates, n_nodes=2000):
    """Return the share of nodes in the largest component after every edge, by edges per node."""
    system = create_system(SYSTEM, n_nodes=n_nodes, candidates=candidates)
    run = Simulation(system, random_state=0).prepare().run(until=int(1.2 * n_nodes))
    return np.arange(int(1.2 * n_nodes) + 1) / n_nodes, run.trajectory_.values('nodes.size').max(axis=1) / n_nodes


# %%
# The giant component
# -------------------
# With one candidate the edges form an Erdős-Rényi graph, whose giant component should grow continuously from half an
# edge per node along `S = 1 - exp(-2 t S)`. The product rule should keep it near zero up to about 0.88 edges per node
# and then jump to most of the nodes.

fig, axis = plt.subplots(figsize=(6, 4))
for candidates, label in ((1, 'one candidate (Erdős-Rényi)'), (2, 'product rule of two candidates')):
    edges, share = largest(candidates)
    axis.plot(edges, share, label=label)
grid = np.linspace(0.51, 1.2, 100)
axis.plot(grid, [brentq(lambda size, t=t: size - 1 + np.exp(-2 * t * size), 1e-9, 1.0) for t in grid], 'k:', label='S')
axis.axvline(0.888, color='gray', linestyle='--', label='t = 0.888')
axis.set_xlabel('edges per node t')
axis.set_ylabel('share of nodes in the largest component')
axis.set_title('The product rule delays and sharpens percolation')
axis.legend()
fig.tight_layout()
plt.show()
