"""
# Random failures and targeted attacks

A scale-free network loses its nodes of the largest degree, one degree class per step. The hubs hold the network
together, so once a sixth of the nodes are gone it falls apart into small pieces, while random failures of as
many nodes leave its giant component almost whole.
"""

# %%
# The targeted attack runs on a Barabási-Albert graph of 300 nodes with two edges per node. The graph view places the
# nodes present with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import networkx as nx
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import NetworkAttackSystem

SYSTEM = 'network_attack'

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
trajectory = simulate(NetworkAttackSystem(), until=17, observers=[observer], random_state=0)


def giant_share(trajectory, frame, size):
    """Return the share of the original nodes in the largest component at a frame."""
    graph = trajectory.get_network(frame).graph()
    return max((len(component) for component in nx.connected_components(graph)), default=0) / size


# %%
# Giant component against removed nodes
# -------------------------------------
# Under a targeted attack the giant component should collapse once the hubs are gone, near a removed share of 0.1,
# while random failures should shrink it only by the nodes they remove, as for a network with no percolation threshold.

size = 1000
fig, axis = plt.subplots(figsize=(6, 4))
for strategy, rate in (('targeted', 0.0), ('random', 0.01)):
    run = simulate(NetworkAttackSystem(n_nodes=size, strategy=strategy, rate=rate), until=60, random_state=0)
    removed = [1 - len(run.get_network(frame).nodes) / size for frame in range(len(run.times))]
    giants = [giant_share(run, frame, size) for frame in range(len(run.times))]
    axis.plot(removed, giants, 'o-', markersize=3, label=strategy)
axis.plot([0, 1], [1, 0], 'k:', label='removed nodes only')
axis.set_xlim(0, 0.5)
axis.set_xlabel('share of nodes removed')
axis.set_ylabel('share of nodes in the giant component')
axis.set_title('Scale-free networks are robust to failures and fragile to attacks')
axis.legend()
fig.tight_layout()
plt.show()
