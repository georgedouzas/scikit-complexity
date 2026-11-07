"""
# Kronecker graphs

The Kronecker graph of two graphs has a node for every pair of their nodes, and two pairs are linked when both of
their parts are linked. Degrees multiply, so two heterogeneous graphs make a product with an even broader spread of
degrees and a mean degree that is the product of theirs.
"""

# %%
# The product of a Barabási-Albert graph of 30 nodes and one of 20 nodes has 600 nodes. The graph view places them
# with its force layout and colors them by degree.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import KroneckerGraphSystem

SYSTEM = 'kronecker_graph'

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
trajectory = simulate(KroneckerGraphSystem(), observers=[observer])

# %%
# Degrees multiply
# ----------------
# The degree of the node `(i, j)` should equal the product of the degrees of `i` and `j`, so the degree distribution of
# the product is the distribution of products of the degrees of the factors (Gros, chapter 1).

first, second = nx.barabasi_albert_graph(30, 2, seed=1), nx.barabasi_albert_graph(20, 1, seed=2)
product = KroneckerGraphSystem(first=first, second=second).prepare().system_.space.graph
expected = [first.degree(pair[0]) * second.degree(pair[1]) for _, pair in product.nodes(data='pair')]
observed = [product.degree(node) for node in product.nodes]
fig, (scatter, tail) = plt.subplots(1, 2, figsize=(10, 4))
scatter.plot(expected, observed, 'o', markersize=3)
scatter.set_xlabel('product of the degrees of the parts')
scatter.set_ylabel('degree in the Kronecker graph')
scatter.set_title(f'Mean degree {np.mean(observed):.2f} = {2 * first.size() / 30 * 2 * second.size() / 20:.2f}')
for graph, label in ((first, 'first'), (second, 'second'), (product, 'product')):
    degrees = np.sort([degree for _, degree in graph.degree])
    tail.loglog(degrees, 1 - np.arange(degrees.size) / degrees.size, 'o', markersize=3, label=label)
tail.set_xlabel('degree k')
tail.set_ylabel('share of nodes of degree k or more')
tail.set_title('A broader spread of degrees')
tail.legend()
fig.tight_layout()
plt.show()
