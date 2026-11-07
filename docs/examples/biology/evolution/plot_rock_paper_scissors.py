"""
# Rock, paper, scissors

On a lattice where rock beats scissors, scissors beat paper, and paper beats rock, no strategy can win: each one
invades its prey and is invaded by its predator. Starting from a random mixture, the three strategies form domains
that chase each other, and each keeps about a third of the lattice.
"""

# %%
# The system is built from its class on a periodic lattice of 60 by 60 sites with random strategies. The observer
# saves every step with the default blueprint, whose primary view is the image of the strategies.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import RockPaperScissorsSystem

SYSTEM = 'rock_paper_scissors'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(RockPaperScissorsSystem(shape=(60, 60), random_state=0), observers=[observer], random_state=0)

# %%
# Coexistence of the three strategies
# -----------------------------------
# The share of each strategy should fluctuate around one third without any of them dying out, while the domains of
# the last snapshot show the cyclic fronts.

strategies = trajectory.lattice('lattice.strategy')
values = np.unique(strategies[0])
shares = np.array([[np.mean(frame == value) for value in values] for frame in strategies])
fig, (snapshot, series) = plt.subplots(1, 2, figsize=(11, 4.5))
snapshot.imshow(strategies[-1], cmap='brg', interpolation='nearest')
snapshot.set_title(f'Strategies at step {trajectory.times[-1]:.0f}')
snapshot.set_xlabel('column')
snapshot.set_ylabel('row')
for value, share in zip(values, shares.T, strict=True):
    series.plot(trajectory.times, share, label=f'strategy {value:g}')
series.axhline(1 / 3, color='black', linestyle='--', label='one third')
series.set_ylim(0, 1)
series.set_xlabel('step')
series.set_ylabel('share of the lattice')
series.set_title('All three strategies coexist')
series.legend()
fig.tight_layout()
plt.show()
