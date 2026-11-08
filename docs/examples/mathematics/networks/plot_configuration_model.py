"""
# Configuration model

Every node gets as many half-links as its degree, and half-links are paired at random. The network has exactly the
degrees asked for and is otherwise as random as possible, and the friends of a node have on average more friends than
it has, the friendship paradox.
"""

# %%
# A thousand nodes with power-law degrees of exponent 2.5 are paired, and the self-loops and multilinks are rewired
# away. The graph view places the nodes with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import ConfigurationModelSystem

SYSTEM = 'configuration_model'

blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/nodes/degree',
        name='degree',
        background=rrb.archetypes.GraphBackground(color=[40, 40, 40]),
        visual_bounds=rrb.VisualBounds2D(x_range=(-400, 400), y_range=(-400, 400)),
        force_link=rrb.archetypes.ForceLink(distance=40.0),
        force_many_body=rrb.archetypes.ForceManyBody(strength=-60.0),
        force_position=rrb.archetypes.ForcePosition(strength=0.05),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = ConfigurationModelSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, observers=[observer])

# %%
# The friendship paradox
# ----------------------
# Following a random link reaches a node in proportion to its degree, so the mean degree of a neighbour should be
# `<k²> / <k>`, above the mean degree `<k>`, and further above it the broader the degrees (Thurner et al., section
# 4.5.1.3). Without rewiring, dropping the self-loops and multilinks lowers the degrees of the hubs a little.

exponents = np.linspace(2.1, 3.5, 8)
means, neighbours = [], []
for exponent in exponents:
    graph = ConfigurationModelSystem(n_nodes=3000, exponent=exponent).prepare().system_.space.graph
    degrees = np.array([degree for _, degree in graph.degree])
    means.append(degrees.mean())
    neighbours.append(np.mean([graph.degree(end) for edge in graph.edges for end in edge]))
graph = system.prepare().system_.space.graph
asked = system.prepare().system_.elements[0].values['degree']
dropped = ConfigurationModelSystem(simple=False).prepare().system_.space.graph
fig, (paradox, degrees_kept) = plt.subplots(1, 2, figsize=(10, 4))
paradox.plot(exponents, means, 'o-', label='mean degree <k>')
paradox.plot(exponents, neighbours, 's-', label='mean degree of a neighbour')
paradox.set_xlabel('exponent of the degrees')
paradox.set_ylabel('degree')
paradox.set_title('Your friends have more friends')
paradox.legend()
degrees_kept.plot(asked, [graph.degree(node) for node in graph.nodes], 'o', markersize=3, label='rewired')
degrees_kept.plot(asked, [dropped.degree(node) for node in dropped.nodes], '.', markersize=3, label='dropped')
degrees_kept.set_xlabel('degree asked for')
degrees_kept.set_ylabel('degree in the graph')
degrees_kept.set_title('Rewiring keeps every degree')
degrees_kept.legend()
fig.tight_layout()
plt.show()
