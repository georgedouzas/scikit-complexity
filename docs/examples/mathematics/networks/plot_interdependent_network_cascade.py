"""
# Cascades in interdependent networks

Two networks share their nodes, each node of one depending on its partner in the other. A node cut off from the giant
component of one network fails, and takes its partner down with it, which cuts off more nodes of the other network.
The cascade either stops at a large mutual giant component or wipes both networks out, an abrupt collapse.
"""

# %%
# Two Erdős-Rényi networks of 2000 nodes and mean degree 4 lose a random fifth of their nodes, and the system logs the
# functional nodes as the cascade alternates between the networks. The view shows the share of functional nodes.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'interdependent_network_cascade'

blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/nodes', name='functional nodes'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
simulation = Simulation(create_system(SYSTEM), dt=1, observers=[observer]).prepare().run(until=40)

# %%
# A discontinuous transition
# --------------------------
# The share of nodes left in the mutual giant component should follow `S = p (1 - exp(-k S))²` while it has a nonzero
# solution, and drop to 0 at once below `p_c = 2.4554 / k`, unlike the continuous percolation of a single network.
# Networks of 1000 nodes cascade for 30 steps at every share.

shares = np.linspace(0.45, 0.95, 11)
survivors = [
    Simulation(create_system(SYSTEM, n_nodes=1000, p=p), dt=30)
    .prepare()
    .run(until=30)
    .trajectory_.values('nodes.active')[-1]
    .mean()
    for p in shares
]
fine = np.linspace(0.4, 1.0, 400)
theory = []
for p in fine:
    share = p
    for _ in range(2000):
        share = p * (1 - np.exp(-4 * share)) ** 2
    theory.append(share)
fig, (cascade, transition) = plt.subplots(1, 2, figsize=(10, 4))
cascade.plot(simulation.trajectory_.times, simulation.trajectory_.values('nodes.active').mean(axis=1), 'o-')
cascade.set_xlabel('step')
cascade.set_ylabel('share of functional nodes')
cascade.set_title('Failures alternate between the networks, p = 0.8')
transition.plot(shares, survivors, 'o', label='simulation')
transition.plot(fine, theory, color='black', label='S = p (1 - exp(-k S))²')
transition.axvline(2.4554 / 4, color='C3', linestyle=':', label='p_c')
transition.set_xlabel('share p left by the attack')
transition.set_ylabel('mutual giant component')
transition.legend()
fig.tight_layout()
plt.show()
