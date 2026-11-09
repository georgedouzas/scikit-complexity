"""
# Forest fire on hexagonal lattices and graphs

Trees grow slowly on the hexagonal cells of a triangular lattice and rare lightning sets them on fire. Fire spreads to
the six neighbours of a burning cell, so fronts move as hexagons, and the forest settles into a state with fires of
every size. The view shows the cells colored by cover, empty, tree, or burning, at the end of the run.
"""

# %%
# The system is built from its class on a periodic 80 x 80 triangular lattice with half of the cells
# forested. The observer saves every second step with the default blueprint, whose primary view shows the cells.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.criticality import HexagonalForestFireSystem

SYSTEM = 'hexagonal_forest_fire'

system = HexagonalForestFireSystem(shape=(80, 80))
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Fire activity and fronts on a graph
# -----------------------------------
# Once the forest has reached its steady density, the trees grow back as fast as fires burn them, and the number of
# burning cells per step on a 64 x 64 lattice should spread over decades, from single cells to fronts that sweep the
# lattice. On a graph, fire moves along edges, so on a fully forested Watts-Strogatz graph without growth a fire lit
# at one node should burn, at every step, exactly the nodes at that graph distance from it.

steady = simulate(HexagonalForestFireSystem(shape=(64, 64)), until=5000, dt=1, random_state=1)
burning = (steady.get_values('forest.cover')[1000:] == 2).sum(axis=1)
histogram = distribution(burning[burning > 0], kind='histogram', bins=15, log=True)
bins, density = histogram.edges, histogram.density
centers = np.sqrt(bins[1:] * bins[:-1])

graph = nx.connected_watts_strogatz_graph(500, 4, 0.1, seed=0)
start = np.ones(500)
start[0] = 2
spread = simulate(HexagonalForestFireSystem(graph=graph, growth=0.0, lightning=0.0, cover=start), until=15, dt=1)
front = (spread.get_values('forest.cover') == 2).sum(axis=1)
distances = np.bincount(list(nx.single_source_shortest_path_length(graph, 0).values()), minlength=len(front))

fig, (fires, fronts) = plt.subplots(1, 2, figsize=(10, 4))
shown = density > 0
fires.loglog(centers[shown], density[shown], 'o')
fires.set_xlabel('burning cells in one step')
fires.set_ylabel('density')
fires.set_title('Fire activity spreads over decades')
fronts.bar(np.arange(len(front)), distances[: len(front)], color='lightgray', label='nodes at distance t')
fronts.plot(np.arange(len(front)), front, 'o', color='firebrick', label='burning nodes at step t')
fronts.set_xlabel('step t')
fronts.set_ylabel('nodes')
fronts.set_title('Fire on a graph moves one edge per step')
fronts.legend()
fig.tight_layout()
plt.show()
