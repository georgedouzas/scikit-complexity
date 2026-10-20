"""
# Ising on a graph

Below its critical temperature the Ising model on a graph orders, with domains of aligned spins that grow until one
color fills the graph. The view shows the spins on the nodes of a honeycomb graph, a lattice with three neighbours per
node, after domains have formed from random spins.
"""

# %%
# The model is created by name from the catalog on the honeycomb graph that NetworkX generates, whose nodes carry
# their positions in the plane, at `T = 1.2 J`, below the critical temperature `2 / ln(2 + sqrt 3) = 1.52 J` of the
# honeycomb lattice. One unit of time is one sweep of Metropolis updates over the classes of a proper coloring of the
# graph. The observer saves every frame with the default blueprint, whose primary view is the graph of the spins.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.experiments import Experiment, Grid
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'graph_ising'

system = create(MODEL, graph=nx.hexagonal_lattice_graph(14, 14), temperature=1.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=60)


# %%
# Magnetization against temperature on a random regular graph
# ------------------------------------------------------------
# A random 4-regular graph is locally a tree, so the Bethe approximation is exact for it, and its critical temperature
# is `T_c = 2 J / ln(k / (k - 2)) = 2 J / ln 2 = 2.89 J` for degree `k = 4`. The mean absolute magnetization of a
# graph of 1000 nodes, averaged over four runs per temperature after a warm-up, should stay near one below `T_c` and
# fall towards zero above it.


def magnetization(trajectory):
    """Return the mean absolute magnetization of the recorded frames."""
    return float(np.abs(trajectory['spins.spin'].mean(axis=1)).mean())


graph = nx.random_regular_graph(4, 1000, seed=0)
temperatures = [1.0, 1.5, 2.0, 2.4, 2.7, 2.9, 3.1, 3.5, 4.5, 6.0]
result = Experiment(
    lambda temperature: create(MODEL, graph=graph, temperature=temperature, initial='up'),
    Grid(temperature=temperatures),
    replicates=4,
    until=300,
    dt=10,
    warmup=100,
    outputs=[magnetization],
    random_state=0,
).run()
rows = sorted(result.summary(), key=lambda row: row['temperature'])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([row['temperature'] for row in rows], [row['magnetization.mean'] for row in rows], 'o-', label='simulation')
ax.axvline(2 / np.log(2), color='black', linestyle='--', label='Bethe T_c = 2 / ln 2')
ax.set_xlabel('temperature T / J')
ax.set_ylabel('mean |m|')
ax.set_title('The random 4-regular graph orders below T_c')
ax.legend()
fig.tight_layout()
plt.show()
