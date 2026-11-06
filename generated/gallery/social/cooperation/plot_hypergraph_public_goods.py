"""
# Public goods game on a hypergraph

Every hyperedge of a random hypergraph is a group that plays the public goods game. A cooperator pays 1 in each of its
groups, the pot is multiplied by the synergy and shared by the members, and every player copies a random neighbour with
the Fermi probability of their payoffs. Cooperation needs a synergy near the sizes of the hyperedges.
"""

# %%
# Four hundred players form hyperedges of sizes 3 and 5, half of them start as cooperators, and the synergy is 4. The
# default blueprint shows the strategies on the network.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'hypergraph_public_goods'
ROUNDS = 100

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
system = create_system(SYSTEM)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=ROUNDS)

# %%
# Cooperation against the synergy
# -------------------------------
# In a well-mixed population defection wins in a group of `m` for a synergy below `m`. The share of cooperators at the
# end of runs should rise with the synergy across the sizes of the hyperedges.


def cooperators(synergy):
    """Return the share of cooperators at the end of a run."""
    run = Simulation(create_system(SYSTEM, synergy=synergy), random_state=0).prepare().run(until=ROUNDS)
    return np.asarray(run.trajectory_.values('players.strategy'))[-1].mean()


synergies = np.linspace(1.0, 7.0, 7)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(synergies, [cooperators(synergy) for synergy in synergies], 'o-')
ax.set_xlabel('synergy')
ax.set_ylabel(f'share of cooperators after {ROUNDS} rounds')
ax.set_title('Synergy sustains cooperation on a hypergraph')
plt.show()
