"""
# Sznajd system

Sznajd-Weron and Sznajd's "united we stand, divided we fall": a pair of neighbours who agree convinces all their
other neighbours. Agreeing pairs grow domains fast, and the population ends in consensus. Which opinion wins is
decided almost entirely by the initial majority, so the probability that opinion 1 wins jumps from 0 to 1 as its
initial share crosses one half.
"""

# %%
# The people sit on a 15 by 15 periodic lattice and half of them start with opinion 1. The nodes keep their lattice
# coordinates as their `pos` attribute, so the graph view draws them on the lattice, on a grey background. The
# blueprint shows the opinions after three steps, when growing domains of agreement compete for the lattice.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import SznajdSystem

SYSTEM = 'sznajd'
STEPS = 30

lattice = nx.convert_node_labels_to_integers(nx.grid_2d_graph(15, 15, periodic=True), label_attribute='pos')
blueprint = rrb.Blueprint(
    rrb.GraphView(origin='/people', name='people', background=rrb.archetypes.GraphBackground(color=[110, 110, 110])),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
system = SznajdSystem(graph=lattice, share=0.5, random_state=2)
trajectory = simulate(system, until=3, observers=[observer], random_state=2)

# %%
# Consensus and the exit probability
# ----------------------------------
# Every run should reach consensus within a few dozen steps. Over ten runs at each initial share, the probability
# that the final consensus is opinion 1, the exit probability, should be close to a step function at one half, as
# Stauffer and co-authors found for the Sznajd system on the square lattice, rather than the straight line of the
# voter system.


def run(share, seed):
    """Return the share of opinion 1 after every step of a run."""
    trajectory = simulate(SznajdSystem(graph=lattice, share=share, random_state=seed), until=STEPS, random_state=seed)
    return np.array([trajectory.agents('people', frame).values['opinion'].mean() for frame in range(STEPS + 1)])


shares = np.linspace(0.3, 0.7, 9)
fig, (series, exit_probability) = plt.subplots(1, 2, figsize=(10, 4))
for seed in range(5):
    series.plot(run(0.5, seed), color='C0', alpha=0.6)
series.set_xlabel('step')
series.set_ylabel('share of opinion 1')
series.set_title('Five runs from a balanced start')
wins = [np.mean([run(share, seed)[-1] > 0.5 for seed in range(10)]) for share in shares]
exit_probability.plot(shares, wins, 'o-', label='Sznajd, 10 runs each')
exit_probability.plot([0, 1], [0, 1], 'k:', label='voter system')
exit_probability.set_xlim(0.25, 0.75)
exit_probability.set_xlabel('initial share of opinion 1')
exit_probability.set_ylabel('probability that opinion 1 wins')
exit_probability.set_title('The initial majority wins')
exit_probability.legend()
fig.tight_layout()
plt.show()
