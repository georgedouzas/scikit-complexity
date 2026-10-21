"""
# Voter variants

Voters on a social network update their opinion from their neighbours: plain voters copy a random neighbour,
threshold voters follow a large enough opposing share of their neighbours, and contrarians do the opposite of what
the others would do. Majority-rule voters quickly agree on one opinion, but a share of contrarians keeps the vote
contested, and enough of them destroy the majority altogether, Galam's contrarian effect.
"""

# %%
# Two hundred voters sit on a random network with eight neighbours on average, 60 percent of them start with opinion
# 1, and they follow the majority rule, `threshold = 0.5`, except for a tenth of contrarians. The nodes carry a spring
# layout as their `pos` attribute, and the blueprint shows the network on a grey background after 50 steps, a large
# majority with a scattering of dissenters.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'voter_variants'
STEPS = 100

graph = nx.gnp_random_graph(200, 8 / 199, seed=0)
nx.set_node_attributes(graph, {node: tuple(10 * xy) for node, xy in nx.spring_layout(graph, seed=0).items()}, 'pos')
blueprint = rrb.Blueprint(
    rrb.GraphView(origin='/voters', name='voters', background=rrb.archetypes.GraphBackground(color=[110, 110, 110])),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, graph=graph, contrarians=0.1, share=0.6, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=50)

# %%
# The contrarian effect
# ---------------------
# The magnetization `m = 2 x - 1`, with `x` the share of opinion 1, measures the majority. Without contrarians the
# majority rule should drive `|m|` to 1 within a few steps. Contrarians should keep it below 1, and as their share
# grows the stationary `|m|` should fall towards the small value of a disordered vote, so that the vote ends contested
# whatever the initial majority.


def magnetization(contrarians, seed, steps=STEPS):
    """Return the magnetization after every step of a run."""
    system = create(MODEL, graph=graph, contrarians=contrarians, share=0.6, random_state=seed)
    trajectory = Simulation(system, random_state=seed).run(until=steps).trajectory_
    return np.array([2 * trajectory.agents('voters', frame).values['opinion'].mean() - 1 for frame in range(steps + 1)])


shares = np.linspace(0.0, 0.5, 11)
fig, (series, sweep) = plt.subplots(1, 2, figsize=(10, 4))
for index, contrarians in enumerate((0.0, 0.1, 0.3)):
    series.plot(magnetization(contrarians, 0), color=f'C{index}', label=f'{contrarians:.0%} contrarians')
series.set_xlabel('step')
series.set_ylabel('magnetization m')
series.set_title('Consensus or a contested vote')
series.legend()
stationary = [np.mean([np.abs(magnetization(share, seed)[50:]).mean() for seed in range(3)]) for share in shares]
sweep.plot(shares, stationary, 'o-')
sweep.axhline(1 / np.sqrt(200), color='grey', linestyle=':', label='random opinions')
sweep.set_xlabel('share of contrarians')
sweep.set_ylabel('stationary |m|')
sweep.set_title('Contrarians destroy the majority')
sweep.legend()
fig.tight_layout()
plt.show()
