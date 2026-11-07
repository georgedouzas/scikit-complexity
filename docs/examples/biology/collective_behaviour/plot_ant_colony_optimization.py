"""
# Ant colony optimization

Ants walk a road network between their nest and a food source, choosing roads by their pheromone and length, and mark
the paths they find. Pheromone gathers fastest on short paths and fades elsewhere, so the colony settles on a short
route, and after two roads of it are cut, explores again and finds a new one.
"""

# %%
# The system is created by name from the catalog on a Delaunay network of 40 points, with the best path cut at round 50.
# The observer saves every round with the default blueprint, whose primary view is the time series of the shortest
# path of every round.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'ant_colony_optimization'

system = create_system(SYSTEM, cut=50)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=100)

# %%
# Convergence and healing
# -----------------------
# The shortest path of a round should fall close to the shortest path of the network, computed by Dijkstra's algorithm,
# jump up when the best route is cut, and settle again near the new shortest path.

search = system.interactions[0]
trajectory = simulation.trajectory_
is_open = trajectory.agents('roads', 100).values['passable']
tau = trajectory.agents('roads', 100).values['tau']
graph = nx.Graph()
for (first, second), length, kept in zip(search.edges, search.lengths, is_open, strict=True):
    graph.add_edge(first, second, weight=length if kept else np.inf)
before = nx.Graph([(a, b, {'weight': length}) for (a, b), length in zip(search.edges, search.lengths, strict=True)])
fig, (series, roads) = plt.subplots(1, 2, figsize=(11, 4.5))
series.plot(trajectory.times, trajectory.values('search.shortest'), label='shortest path of a round')
series.axhline(nx.shortest_path_length(before, search.start, search.target, weight='weight'), color='black', ls='--')
series.axhline(nx.shortest_path_length(graph, search.start, search.target, weight='weight'), color='gray', ls=':')
series.axvline(50, color='tab:red', alpha=0.5, label='cut')
series.set_xlabel('round')
series.set_ylabel('path length')
series.set_title('Colony converges, then heals the cut')
series.legend()
points = np.random.default_rng(0).random((40, 2))
for (first, second), level, kept in zip(search.edges, tau, is_open, strict=True):
    style = '-' if kept else ':'
    roads.plot(*points[[first, second]].T, style, color='tab:orange', lw=0.5 + 4 * level / max(tau.max(), 1e-9))
roads.plot(*points[search.start], 's', color='black')
roads.plot(*points[search.target], 'o', color='black')
roads.set_title('Pheromone at the end, cut roads dotted')
roads.set_aspect('equal')
fig.tight_layout()
plt.show()
