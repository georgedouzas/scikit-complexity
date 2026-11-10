"""
# Popularity-similarity network

Every step a node joins at a random angle and links to nodes close to it in hyperbolic space, where old nodes sit near
the centre and nodes at similar angles sit close together. Popularity and similarity together make a network with hubs
and tight clusters, and blocks of consecutive angles merge into a network that keeps its degree exponent.
"""

# %%
# The network starts complete on three nodes and grows to 300 nodes, logged every second step. The graph view places
# the nodes with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import renormalization
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import PopularitySimilarityNetworkSystem

SYSTEM = 'popularity_similarity_network'

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
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(PopularitySimilarityNetworkSystem(), dt=2.0, observers=[observer], random_state=0)

# %%
# Clustering against the temperature
# ----------------------------------
# The temperature sets how far links reach, so the clustering falls as it rises while the number of links stays the
# same.

temperatures = np.linspace(0.1, 0.9, 5)
clustering = []
for temperature in temperatures:
    run = simulate(PopularitySimilarityNetworkSystem(n_nodes=1000, temperature=temperature), until=997, random_state=1)
    clustering.append(nx.average_clustering(run.get_network(-1).get_graph()))
fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(temperatures, clustering, 'o-')
axis.set_xlabel('temperature')
axis.set_ylabel('average clustering')
axis.set_title('A higher temperature lowers the clustering')
fig.tight_layout()
plt.show()

# %%
# The network across scales
# -------------------------
# `renormalization(kind='geometric')` merges pairs of consecutive nodes by angle three times. The tail of the
# degree distribution keeps its slope at every scale, and the clustering stays high.

run = simulate(PopularitySimilarityNetworkSystem(n_nodes=2000), until=1997, dt=1997.0, random_state=0)
scales = renormalization(run, kind='geometric', steps=3, beta=5.0)
fig, axis = plt.subplots(figsize=(6, 4))
for level, degrees in enumerate(scales.degrees):
    values = np.sort(degrees)[::-1]
    shares = np.arange(1, values.size + 1) / values.size
    label = f'{values.size} nodes: exponent {scales.exponents[level]:.2f}, clustering {scales.clustering[level]:.2f}'
    axis.loglog(values, shares, label=label)
axis.set_xlabel('degree k')
axis.set_ylabel('share of nodes with degree at least k')
axis.set_title('The degree distribution at every scale')
axis.legend()
fig.tight_layout()
plt.show()
