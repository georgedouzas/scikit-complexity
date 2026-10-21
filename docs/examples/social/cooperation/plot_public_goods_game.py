"""
# Public goods game

Players on a lattice contribute to the public goods of five overlapping groups, each formed by a player and its four
neighbours, and every group multiplies its contributions by a synergy factor and shares them equally. Free riders
always earn more within a group, yet cooperators survive in compact clusters once the synergy is high enough, and the
share of cooperators rises from zero to one over a window of synergies.
"""

# %%
# On a 50 by 50 periodic lattice, half the players start as cooperators, the synergy factor is `r = 4.5`, and players
# copy a random neighbour with the Fermi probability at the noise `K = 0.5`. The blueprint shows the lattice,
# cooperators in yellow and defectors in purple, after 200 rounds, when cooperators hold most of the lattice and the
# defectors survive in scattered patches.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'public_goods_game'
ROUNDS = 200

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, synergy=4.5, noise=0.5, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=ROUNDS)

# %%
# Cooperation against the synergy
# -------------------------------
# In a well-mixed population defection dominates for every synergy below the group size 5. On the lattice, Szolnoki,
# Perc, and Szabo found that cooperators survive above `r ~ 3.74` and take the whole lattice above `r ~ 5.49` at this
# noise. The share of cooperators at the end of runs at a range of synergies should rise between these thresholds,
# blurred by the finite lattice and run length.


def cooperators(synergy):
    """Return the share of cooperators after every round."""
    run = Simulation(create(MODEL, synergy=synergy, noise=0.5, random_state=0), random_state=0).run(until=ROUNDS)
    return run.trajectory_['players.strategy'].reshape(ROUNDS + 1, -1).mean(axis=1)


synergies = np.linspace(3.0, 6.5, 8)
fig, (series, sweep) = plt.subplots(1, 2, figsize=(10, 4))
finals = []
for synergy in synergies:
    shares = cooperators(synergy)
    finals.append(shares[-20:].mean())
    series.plot(shares, label=f'r = {synergy:.1f}')
series.set_xlabel('round')
series.set_ylabel('share of cooperators')
series.set_title('Cooperation over time')
series.legend(fontsize=7, ncol=2)
sweep.plot(synergies, finals, 'o-')
for threshold in (3.74, 5.49):
    sweep.axvline(threshold, color='grey', linestyle='--', lw=0.8)
sweep.set_xlabel('synergy factor r')
sweep.set_ylabel('share of cooperators, last 20 rounds')
sweep.set_title('Cooperators survive above r ~ 3.74')
fig.tight_layout()
plt.show()
