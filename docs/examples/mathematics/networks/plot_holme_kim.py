"""
# Holme-Kim growth

Every step a node joins with one edge by preferential attachment and then adds two edges that close triangles with
the neighbours of its first contact. The network keeps the hubs of Barabási-Albert growth, bright by their degree,
and gains tight local clusters that preferential attachment alone does not make.
"""

# %%
# The network starts complete on four nodes and grows to 304 nodes, logged every second step, with triad formation
# probability 0.8. The graph view places the nodes with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import as_graph
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'holme_kim'

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
simulation = Simulation(system, dt=2.0, observers=[observer], random_state=0).run(until=300)

# %%
# Clustering against triad formation
# ----------------------------------
# The clustering coefficient should grow with the triad formation probability and stay far above the clustering of
# Barabási-Albert growth with the same three edges per node, which is near zero, while the number of edges stays the
# same.

triads = np.linspace(0.0, 1.0, 6)
clustering = []
for triad in triads:
    run = Simulation(create(MODEL, n_nodes=1004, triad=triad), dt=500.0, random_state=1).run(until=1000)
    clustering.append(nx.average_clustering(as_graph(run.trajectory_.network(-1))))
baseline = Simulation(create('barabasi_albert', n_nodes=1003, edges=3), dt=500.0, random_state=1).run(until=1000)
fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(triads, clustering, 'o-', label='Holme-Kim')
axis.axhline(
    nx.average_clustering(as_graph(baseline.trajectory_.network(-1))),
    color='k',
    ls='--',
    label='Barabási-Albert',
)
axis.set_xlabel('triad formation probability')
axis.set_ylabel('average clustering')
axis.set_title('Triad formation keeps the clustering of a scale-free network high')
axis.legend()
fig.tight_layout()
plt.show()
