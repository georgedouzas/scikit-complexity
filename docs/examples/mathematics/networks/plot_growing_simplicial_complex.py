"""
# Growing simplicial complex

The complex starts as one triangle, and every step a node joins and forms a triangle with the ends of an edge drawn
uniformly. A node gains triangles in proportion to its degree, so hubs emerge and the degrees follow the power law of
preferential attachment.
"""

# %%
# The complex grows for 300 steps, logged every second step. The graph view places the nodes with its force layout and
# colors them by degree.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.analysis import DistributionAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'growing_simplicial_complex'

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
simulation = Simulation(system, dt=2.0, observers=[observer], random_state=0).check().prepare().run(until=300)

# %%
# Degree tail
# -----------
# The degrees should follow `P(k) ~ k^-3`, so the share of nodes of degree at least `k` should fall as `k^-2`.

run = Simulation(create(MODEL, n_nodes=3003), dt=3000.0, random_state=0).check().prepare().run(until=3000)
snapshot = run.trajectory_.network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
ccdf = DistributionAnalysis('ccdf').check().prepare(degrees).run()
values, shares = ccdf.support_, ccdf.survival_
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(values, shares, 'o', markersize=3, label='simulation')
ax.loglog(values, shares[0] * (values / values[0]) ** -2.0, '--', color='black', label='k^-2')
ax.set_xlabel('degree k')
ax.set_ylabel('P(K >= k)')
ax.set_title('Attaching triangles along edges grows hubs')
ax.legend()
plt.show()
