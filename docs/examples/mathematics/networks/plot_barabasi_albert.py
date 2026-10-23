"""
# Barabási-Albert growth

Every step a node joins with two edges to nodes drawn with probability proportional to their degree. The rich get
richer: early nodes grow into hubs, colored bright by their degree, while most nodes keep two or three edges, so the
degrees follow the power law `P(k) ~ k^-3`.
"""

# %%
# The network starts complete on three nodes and grows to 303 nodes, logged every second step. The graph view places
# the nodes with its force layout, and the colors run from the lowest degree (dark) to the highest (yellow).

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import ccdf
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'barabasi_albert'

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
# Degree distribution
# -------------------
# The complementary cumulative distribution of the degrees should fall as `k^-2`, the integral of `P(k) ~ k^-3`, in a
# larger network grown with the same law, with the mean-field prediction `P(K >= k) = m^2 / k^2` for `m = 2` edges.

grown = Simulation(create(MODEL, n_nodes=5003), dt=1000.0, random_state=0).run(until=5000).trajectory_
snapshot = grown.network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
values, shares = ccdf(degrees)
fig, axis = plt.subplots(figsize=(6, 4))
axis.loglog(values, shares, 'o', markersize=3, label='grown network')
axis.loglog(values, np.minimum(1.0, 4.0 / values**2), 'k--', label=r'$m^2 / k^2$')
axis.set_xlabel('degree k')
axis.set_ylabel('P(K >= k)')
axis.set_title(f'Degrees of {snapshot.nodes.size} nodes follow a power law')
axis.legend()
fig.tight_layout()
plt.show()
