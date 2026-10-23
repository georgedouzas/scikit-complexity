"""
# Erdős-Rényi and Watts-Strogatz graphs

An Erdős-Rényi graph is built one step of random edges at a time on 200 nodes. Isolated nodes and small trees merge
into one giant component, colored by degree, once the mean degree passes 1.
"""

# %%
# The graph starts empty and reaches mean degree 4 after 100 steps. The graph view places the nodes with its force
# layout and colors them by degree.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import as_graph
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'random_graph_ensembles'

system = create(MODEL)
# The graph view lays out the nodes of a changing network by forces, which these settings keep compact and centred.
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/nodes/degree',
        name='degree',
        background=rrb.archetypes.GraphBackground(color=[40, 40, 40]),
        visual_bounds=rrb.VisualBounds2D(x_range=(-330, 330), y_range=(-330, 330)),
        force_link=rrb.archetypes.ForceLink(distance=40.0),
        force_many_body=rrb.archetypes.ForceManyBody(strength=-60.0),
        force_position=rrb.archetypes.ForcePosition(strength=0.05),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=100)
trajectory = simulation.trajectory_

# %%
# Giant component and small worlds
# --------------------------------
# The giant component of the Erdős-Rényi graph should follow `S = 1 - exp(-c S)` in the mean degree `c`, zero below 1.
# For the Watts-Strogatz construction, the mean path length should drop with the rewired share long before the
# clustering does, the small-world regime.

degrees, giants = [], []
for frame in range(len(trajectory)):
    graph = as_graph(trajectory.network(frame))
    degrees.append(2 * graph.number_of_edges() / 200)
    giants.append(max(len(component) for component in nx.connected_components(graph)) / 200)
mean = np.linspace(0.01, 4, 200)
theory = np.zeros_like(mean)
for _ in range(200):
    theory = 1 - np.exp(-mean * np.maximum(theory, 1e-3))
theory[mean < 1] = 0
shares = np.logspace(-3, 0, 8)
ring = nx.watts_strogatz_graph(300, 6, 0.0)
paths, clustering = [], []
for share in shares:
    run = Simulation(create(MODEL, n_nodes=300, ensemble='watts_strogatz', mean_degree=6, rewiring=share, steps=20))
    graph = as_graph(run.run(until=20).trajectory_.network(-1))
    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    paths.append(nx.average_shortest_path_length(graph) / nx.average_shortest_path_length(ring))
    clustering.append(nx.average_clustering(graph) / nx.average_clustering(ring))
fig, (giant, small) = plt.subplots(1, 2, figsize=(11, 4))
giant.plot(degrees, giants, 'o', markersize=3, label='construction')
giant.plot(mean, theory, 'k--', label=r'$S = 1 - e^{-cS}$')
giant.set_xlabel('mean degree c')
giant.set_ylabel('giant component share S')
giant.set_title('Erdős-Rényi giant component')
giant.legend()
small.semilogx(shares, paths, 'o-', label='L(p) / L(0)')
small.semilogx(shares, clustering, 's-', label='C(p) / C(0)')
small.set_xlabel('rewired share p')
small.set_title('Watts-Strogatz small world')
small.legend()
fig.tight_layout()
plt.show()
