"""
# Metapopulation SIR

An epidemic that starts in one city of a network spreads to the others with the travellers: the hubs, with many
connections, are reached first, and the less connected cities peak later, one wave after another.
"""

# %%
# The example builds `MetapopulationSystem` on its default Barabási-Albert network of 30 cities, with the infection
# starting in the first. The observer saves every day with the default blueprint, whose primary view is the time series
# of the subpopulations.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import MetapopulationSystem

SYSTEM = 'metapopulation'

graph = nx.barabasi_albert_graph(30, 2, seed=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(MetapopulationSystem(graph=graph), observers=[observer])

# %%
# Arrival time against distance
# -----------------------------
# The infection hops one edge at a time, so the day each city reaches its epidemic peak should grow with its distance
# on the network from the first city.

infected = trajectory.get_values('cities.I')
peaks = trajectory.times[np.argmax(infected, axis=0)]
distance = nx.single_source_shortest_path_length(graph, 0)
hops = np.array([distance[node] for node in range(graph.number_of_nodes())])
fig, (curves, arrival) = plt.subplots(1, 2, figsize=(11, 4))
curves.plot(trajectory.times, infected, linewidth=0.8)
curves.set_xlabel('day')
curves.set_ylabel('infected per city')
curves.set_title('Waves spread from city to city')
arrival.plot(hops + np.random.default_rng(0).uniform(-0.1, 0.1, hops.size), peaks, 'o')
arrival.set_xlabel('distance from the first city (edges)')
arrival.set_ylabel('day of the epidemic peak')
arrival.set_title('Farther cities peak later')
fig.tight_layout()
plt.show()
