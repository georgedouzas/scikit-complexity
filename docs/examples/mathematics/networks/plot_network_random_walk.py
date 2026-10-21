"""
# Random walk on a network

Walkers hop from node to node of Zachary's karate club along random links. Whatever their start, they soon crowd onto
the hubs: on a connected graph a random walker spends a share of its time on every node proportional to the node's
degree.
"""

# %%
# Two hundred walkers start at random nodes and take 200 steps. After the run the number of walkers on every
# node at every step is logged as the size and colour of the nodes of the club, and the blueprint shows this graph as
# the primary view.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'network_random_walk'

graph = nx.karate_club_graph()
blueprint = rrb.Blueprint(
    rrb.GraphView(origin='/club', name='walkers on every node'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, graph=graph, n_walkers=200), dt=1, observers=[observer], random_state=0)
simulation.run(until=200)
trajectory = simulation.trajectory_
layout = nx.spring_layout(graph, seed=1)
positions = np.array([layout[node] for node in graph.nodes]) * 300
observer.recording_.log('club', rr.GraphEdges(list(graph.edges), graph_type='undirected'), static=True)
for frame, time in enumerate(trajectory.times):
    located = np.asarray(trajectory.agents('walkers', frame).locations).ravel().astype(int)
    crowd = np.bincount(located, minlength=graph.number_of_nodes())
    observer.recording_.set_time('sim_time', duration=time)
    observer.recording_.log(
        'club',
        rr.GraphNodes(
            list(graph.nodes),
            positions=positions,
            colors=plt.get_cmap('plasma')(np.minimum(crowd / 25, 1))[:, :3],
            radii=3 + 1.2 * crowd,
            show_labels=False,
        ),
    )
observer.recording_.flush()

# %%
# Occupation proportional to degree
# ---------------------------------
# The stationary distribution of a random walk on an undirected graph is `π_i = k_i / 2m`, for the degree `k_i` and
# `m` links. The share of walker-steps on each node, counted after 20 steps of transient, should match it, so the
# hubs 0 and 33 should hold about 5 and 6 times the share of a node of degree 3.

nodes = np.concatenate(
    [np.asarray(trajectory.agents('walkers', frame).locations).ravel() for frame in range(20, len(trajectory.times))],
)
share = np.bincount(nodes.astype(int), minlength=graph.number_of_nodes()) / nodes.size
degree = np.array([graph.degree(node) for node in graph.nodes])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(degree / (2 * graph.number_of_edges()), share, 'o', label='nodes')
line = np.array([0, degree.max() / (2 * graph.number_of_edges())])
ax.plot(line, line, 'k--', label=r'$\pi_i = k_i / 2m$')
ax.set_xlabel(r'degree share $k_i / 2m$')
ax.set_ylabel('share of walker-steps')
ax.set_title('Walkers visit nodes in proportion to their degree')
ax.legend()
plt.show()
