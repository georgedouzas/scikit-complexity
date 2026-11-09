"""
# Graphlet arrival growth

Every step a graphlet of three nodes, a centre with two peripheral nodes, arrives and merges one of its nodes with an
existing node drawn in proportion to its degree. Merging by a peripheral node hangs a short branch, and merging by the
centre adds two leaves, so the tree grows hubs whose degree tail steepens as branches become more common.
"""

# %%
# The tree starts from one edge and grows to 602 nodes with `alpha = 0.3`, logged every second step. The graph view
# places the nodes with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import GraphletArrivalSystem

SYSTEM = 'graphlet_arrival'

# The graph view lays out the nodes of a changing network by forces, which these settings keep compact and centred.
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/nodes/degree',
        name='degree',
        background=rrb.archetypes.GraphBackground(color=[40, 40, 40]),
        visual_bounds=rrb.VisualBounds2D(x_range=(-330, 330), y_range=(-330, 330)),
        force_link=rrb.archetypes.ForceLink(distance=30.0),
        force_many_body=rrb.archetypes.ForceManyBody(strength=-40.0),
        force_position=rrb.archetypes.ForcePosition(strength=0.05),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(GraphletArrivalSystem(), dt=2.0, observers=[observer], random_state=0)

# %%
# Shares of small degrees
# -----------------------
# The shares of the nodes of degree 1 and 2 should follow `2 (2 - alpha) / 5` and `alpha (7 - alpha) / 15` of Filkov
# et al.: attaching by the centre makes more leaves, and attaching by a peripheral node more nodes of degree 2.

alphas = np.linspace(0.0, 1.0, 11)
ones, twos = [], []
for alpha in alphas:
    snapshot = simulate(GraphletArrivalSystem(n_nodes=2002, alpha=alpha), until=1000, random_state=0).get_network(-1)
    degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
    ones.append(np.mean(degrees == 1))
    twos.append(np.mean(degrees == 2))
fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(alphas, ones, 'o', color='C0', label='degree 1')
axis.plot(alphas, 2 * (2 - alphas) / 5, color='C0')
axis.plot(alphas, twos, 's', color='C1', label='degree 2')
axis.plot(alphas, alphas * (7 - alphas) / 15, color='C1')
axis.set_xlabel('alpha, the share of peripheral attachments')
axis.set_ylabel('share of nodes')
axis.set_title('Simulated shares (marks) and theory (lines)')
axis.legend()
fig.tight_layout()
plt.show()
