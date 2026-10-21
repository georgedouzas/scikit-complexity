"""
# Binary ant decision

Ants that reach two equal branches to a food source follow the marks of earlier ants, and every ant reinforces the
branch it takes. A random early lead on one branch is amplified until most of the colony uses it, the symmetry
breaking of the double bridge experiment.
"""

# %%
# The model runs as a stochastic process of single ants arriving at the fork. The observer saves every step with the
# default blueprint, whose primary view is the time series of the ants on each branch.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'binary_ant_decision'

system = create(MODEL, stochastic=True)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0).run(until=200)

# %%
# Symmetry breaking across colonies
# ---------------------------------
# Both branches are equal, so the chosen branch differs from colony to colony, but each colony should end with a
# majority on one branch: the share of the left branch should split into two humps on either side of one half, while
# the deterministic mode stays exactly at one half.

shares = []
for seed in range(40):
    trajectory = Simulation(create(MODEL, stochastic=True), dt=1.0, random_state=seed).run(until=200).trajectory_
    left, right = trajectory['left.n'][-1], trajectory['right.n'][-1]
    shares.append(left / (left + right))
deterministic = Simulation(create(MODEL, stochastic=False), dt=1.0).run(until=200).trajectory_
even = deterministic['left.n'][-1] / (deterministic['left.n'][-1] + deterministic['right.n'][-1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(shares, bins=np.linspace(0, 1, 21))
ax.axvline(even, color='black', linestyle='--', label=f'deterministic mode: {even:.2f}')
ax.set_xlabel('share of ants on the left branch after 200 steps')
ax.set_ylabel('colonies')
ax.set_title('Forty colonies each pick one branch')
ax.legend()
plt.show()
