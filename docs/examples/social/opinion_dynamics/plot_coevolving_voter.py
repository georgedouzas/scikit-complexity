"""
# Coevolving voter

Voters with random opinions on a random graph either copy a disagreeing neighbour or move the edge to a voter who
agrees. With rewiring probability 0.6, above the fragmentation transition near 0.38, the disagreeing edges vanish and
the network splits into groups of purple and yellow voters with no edge between them.
"""

# %%
# The 200 voters start on an Erdős-Rényi graph of mean degree 4. The graph view places the voters with its force
# layout and colors them by opinion.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'coevolving_voter'

system = create(MODEL, rewiring=0.6)
# The graph view lays out the nodes of a changing network by forces, which these settings keep compact and centred.
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/voters/opinion',
        name='opinion',
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
simulation = Simulation(system, dt=2.0, observers=[observer], random_state=0).run(until=200)


def discordant(trajectory, frame):
    """Return the share of the edges of a frame whose ends disagree."""
    snapshot = trajectory.network(frame)
    opinions = trajectory['voters.opinion'][frame]
    return float(np.mean(opinions[snapshot.sources] != opinions[snapshot.targets]))


# %%
# Density of disagreeing edges
# ----------------------------
# Below the transition the density of disagreeing edges should stay on a plateau for a long time before a finite
# network reaches consensus, and above it the density should fall to zero within a few steps, as the network
# fragments.

fig, axis = plt.subplots(figsize=(6, 4))
for rewiring in (0.1, 0.3, 0.6, 0.9):
    run = Simulation(create(MODEL, n_voters=400, rewiring=rewiring), dt=5.0, random_state=1).run(until=300)
    trajectory = run.trajectory_
    axis.plot(
        trajectory.times,
        [discordant(trajectory, frame) for frame in range(len(trajectory))],
        label=f'{rewiring}',
    )
axis.set_xlabel('step')
axis.set_ylabel('share of disagreeing edges')
axis.set_title('Fast rewiring fragments the network')
axis.legend(title='rewiring')
fig.tight_layout()
plt.show()
