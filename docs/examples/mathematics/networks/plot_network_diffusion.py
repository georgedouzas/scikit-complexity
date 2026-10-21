"""
# Diffusion on a network

A unit mass placed on one node of Zachary's karate club leaks along the links: it floods the neighbours of the source
first, reaches the far side of the club last, and relaxes to the same share `1 / n` on every node at a rate set by the
smallest nonzero eigenvalue of the graph Laplacian.
"""

# %%
# The mass starts on node 0 and diffuses with `D = 1` for 0.6 time units. The model logs the contents as time series;
# after the run the contents at every frame are logged again as the colours of the nodes of the graph, on a
# logarithmic scale from `10^-3` (dark) to 1 (yellow), and the blueprint puts this graph first, as the primary view.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'network_diffusion'

graph = nx.karate_club_graph()
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.GraphView(origin='/club', name='content of every node'),
        rrb.TimeSeriesView(origin='/nodes', name='contents'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, graph=graph), dt=0.01, observers=[observer]).run(until=0.6)
trajectory = simulation.trajectory_
contents = trajectory['nodes.u']
layout = nx.spring_layout(graph, seed=1)
positions = np.array([layout[node] for node in graph.nodes]) * 300
observer.recording_.log('club', rr.GraphEdges(list(graph.edges), graph_type='undirected'), static=True)
for time, row in zip(trajectory.times, contents, strict=True):
    observer.recording_.set_time('sim_time', duration=time)
    shades = plt.get_cmap('viridis')(np.clip((np.log10(np.maximum(row, 1e-3)) + 3) / 3, 0, 1))[:, :3]
    observer.recording_.log(
        'club',
        rr.GraphNodes(list(graph.nodes), positions=positions, colors=shades, radii=12, show_labels=False),
    )
observer.recording_.flush()

# %%
# Relaxation to the uniform state
# -------------------------------
# The total mass should stay 1, and the distance of the contents from the uniform share `1 / n` should decay at long
# times as `exp(-D λ2 t)`, with `λ2` the algebraic connectivity of the club, the smallest nonzero eigenvalue of its
# Laplacian.

run = Simulation(create(MODEL, graph=graph), dt=0.1).run(until=20).trajectory_
distance = np.linalg.norm(run['nodes.u'] - 1 / graph.number_of_nodes(), axis=1)
connectivity = nx.algebraic_connectivity(graph, weight=None, seed=0)
fig, (decay, by_distance) = plt.subplots(1, 2, figsize=(11, 4))
decay.semilogy(run.times, distance, label='simulated')
decay.semilogy(run.times, distance[50] * np.exp(-connectivity * (run.times - 5)), 'k--', label=r'$e^{-\lambda_2 t}$')
decay.set_xlabel('time')
decay.set_ylabel('distance from 1/n')
decay.set_title(f'Relaxation at λ2 = {connectivity:.3f}; mass drift {np.ptp(run["nodes.u"].sum(axis=1)):.0e}')
decay.legend()
hops = nx.single_source_shortest_path_length(graph, 0)
for hop in sorted(set(hops.values())):
    members = [node for node, length in hops.items() if length == hop]
    by_distance.semilogy(trajectory.times, contents[:, members].mean(axis=1), label=f'{hop} links from the source')
by_distance.set_xlabel('time')
by_distance.set_ylabel('mean content')
by_distance.set_title('Nodes farther away fill later')
by_distance.legend()
fig.tight_layout()
plt.show()
