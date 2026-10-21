"""
# Cognitive consistency

Agents talk about two topics and settle what they hear inside their own belief network, which links the topics to an
attitude and seeks consistency between them. Conversation pulls the valences of the topics together, and each mind
turns the shared valences into an attitude, so the population converges on the attitude of its initial majority
when everyone talks to everyone, while talk among neighbours lets local camps survive with a smooth front between
them.
"""

# %%
# Forty thinkers sit on a ring, each talking to its two nearest neighbours on either side, and 70 percent of them start
# in the positive state. The nodes carry their position on a circle as their `pos` attribute, and the blueprint shows
# the attitudes after 100 steps on a grey background, moderate positive attitudes along most of the ring and a
# surviving negative arc, joined by a smooth gradient.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cognitive_consistency'
N_THINKERS, STEPS = 40, 100

ring = nx.watts_strogatz_graph(N_THINKERS, 4, 0.0, seed=0)
angles = 2 * np.pi * np.arange(N_THINKERS) / N_THINKERS
nx.set_node_attributes(
    ring,
    {node: (10 * np.cos(angle), 10 * np.sin(angle)) for node, angle in enumerate(angles)},
    'pos',
)
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/thinkers',
        name='thinkers',
        background=rrb.archetypes.GraphBackground(color=[110, 110, 110]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, graph=ring, positive=0.7, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=STEPS)

# %%
# Convergence on the majority
# ---------------------------
# On the complete graph the share of thinkers with a positive attitude should move to 1 when 70 percent start positive
# and to 0 when 30 percent do, within a few dozen steps, and the spread of the attitudes should vanish. On the ring,
# where agreement has to travel from neighbour to neighbour, the minority camp can survive as an arc, and the spread
# should level off above zero when it does.


def attitudes(graph, positive, seed=0):
    """Return the attitudes of the thinkers at every step."""
    run = Simulation(create(MODEL, graph=graph, positive=positive, random_state=seed), random_state=seed)
    trajectory = run.run(until=STEPS).trajectory_
    return np.array([trajectory.agents('thinkers', frame).values['attitude'] for frame in range(STEPS + 1)])


fig, (shares, spread) = plt.subplots(1, 2, figsize=(10, 4))
for graph, name, style in ((None, 'complete graph', '-'), (ring, 'ring', '--')):
    for positive, color in ((0.7, 'C1'), (0.3, 'C0')):
        history = attitudes(graph, positive)
        label = f'{name}, {positive:.0%} positive'
        shares.plot((history > 0).mean(axis=1), style, color=color, label=label)
        spread.plot(history.std(axis=1), style, color=color, label=label)
shares.set_xlabel('step')
shares.set_ylabel('share with a positive attitude')
shares.set_title('The initial majority wins')
shares.legend(fontsize=8)
spread.set_xlabel('step')
spread.set_ylabel('standard deviation of the attitudes')
spread.set_title('Attitudes converge')
spread.legend(fontsize=8)
fig.tight_layout()
plt.show()
