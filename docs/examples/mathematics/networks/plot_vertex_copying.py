"""
# Vertex-copying growth

Every step a node joins with three edges, each copying an edge of a random prototype with probability 0.8 and joining
a random node otherwise. A node is copied in proportion to its number of neighbours, so hubs emerge, bright by their
degree, without any node reading degrees.
"""

# %%
# The network starts complete on four nodes and grows to 304 nodes, logged every second step. The graph view places
# the nodes with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import ccdf
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'vertex_copying'

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
# Degree tails
# ------------
# Copying should give a heavy tail of degrees that widens as the copying share grows, while joining random nodes alone,
# `randomness = 1`, should give the thin, exponential tail of random attachment.

fig, axis = plt.subplots(figsize=(6, 4))
for randomness in (1.0, 0.5, 0.1):
    run = Simulation(create(MODEL, n_nodes=3004, randomness=randomness), dt=1000.0, random_state=0).run(until=3000)
    snapshot = run.trajectory_.network(-1)
    values, shares = ccdf(np.bincount(np.concatenate([snapshot.sources, snapshot.targets])))
    axis.loglog(values, shares, 'o', markersize=3, label=f'randomness {randomness}')
axis.set_xlabel('degree k')
axis.set_ylabel('P(K >= k)')
axis.set_title('Copying edges grows hubs')
axis.legend()
fig.tight_layout()
plt.show()
