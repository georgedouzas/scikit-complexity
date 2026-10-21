"""
# PageRank

Iterating the random surfer's update from uniform ranks on Zachary's karate club concentrates the rank on the two
hubs, the instructor and the administrator, within a few steps, and converges geometrically to the PageRank of the
graph, at a rate set by the damping factor.
"""

# %%
# The ranks start uniform and are updated 40 times with the damping `d = 0.85`. The model logs the ranks as time
# series; after the run the ranks at every step are logged again as the sizes of the nodes of the graph, and the
# blueprint puts this graph first, as the primary view.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pagerank'

graph = nx.karate_club_graph()
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.GraphView(origin='/club', name='rank of every node'),
        rrb.TimeSeriesView(origin='/pages', name='ranks'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, graph=graph), dt=1, observers=[observer]).run(until=40)
trajectory = simulation.trajectory_
ranks = trajectory['pages.rank']
layout = nx.spring_layout(graph, seed=1)
positions = np.array([layout[node] for node in graph.nodes]) * 300
observer.recording_.log('club', rr.GraphEdges(list(graph.edges), graph_type='undirected'), static=True)
for time, row in zip(trajectory.times, ranks, strict=True):
    observer.recording_.set_time('sim_time', duration=time)
    shades = plt.get_cmap('plasma')(row / ranks.max())[:, :3]
    observer.recording_.log(
        'club',
        rr.GraphNodes(list(graph.nodes), positions=positions, colors=shades, radii=200 * row, show_labels=False),
    )
observer.recording_.flush()

# %%
# Convergence to PageRank
# -----------------------
# The ranks should stay normalized and converge to NetworkX's PageRank of the unweighted graph, with the error falling
# by the factor `d = 0.85` per step at worst, and the final ranks should grow with the degree of the nodes.

exact = nx.pagerank(graph, alpha=0.85, weight=None, tol=1e-12)
exact = np.array([exact[node] for node in graph.nodes])
error = np.abs(ranks - exact).sum(axis=1)
fig, (convergence, by_degree) = plt.subplots(1, 2, figsize=(11, 4))
convergence.semilogy(trajectory.times, error, 'o-', markersize=3, label='simulated')
convergence.semilogy(trajectory.times, error[0] * 0.85**trajectory.times, 'k--', label='$0.85^t$')
convergence.set_xlabel('step')
convergence.set_ylabel('L1 distance to PageRank')
convergence.set_title(f'Geometric convergence; total rank {ranks[-1].sum():.6f}')
convergence.legend()
degree = np.array([graph.degree(node) for node in graph.nodes])
by_degree.plot(degree, ranks[-1], 'o', label='simulated')
by_degree.plot(degree, exact, 'k.', markersize=4, label='networkx.pagerank')
by_degree.set_xlabel('degree')
by_degree.set_ylabel('rank')
by_degree.set_title('Hubs collect the rank')
by_degree.legend()
fig.tight_layout()
plt.show()
