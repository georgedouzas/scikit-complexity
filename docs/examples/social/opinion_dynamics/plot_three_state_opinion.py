"""
# Three-state opinion with undecided agents

Agents on a network are against, undecided, or for, and redraw their opinion from the sum of the opinions of their
neighbours with logistic probabilities. Half of the draws are undecided, and the agents for and against order into a
majority only when the sensitivity to the neighbours is large enough.
"""

# %%
# Two hundred agents with four neighbours each redraw their opinions with the sensitivity `beta = 2`. The default
# blueprint shows the opinions on the network.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import ThreeStateOpinionSystem

SYSTEM = 'three_state_opinion'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(ThreeStateOpinionSystem(), observers=[observer], random_state=0)

# %%
# Order with the sensitivity
# --------------------------
# In mean field the mean opinion solves `<x> = tanh(beta q <x> / 2) / 2`, which leaves zero above `beta_c = 4 / q = 1`.
# On the network the mean opinion should stay near zero at small sensitivity and grow with it, below the mean-field
# curve, while the share of undecided agents stays at one half.

sensitivities = [0.5, 1.0, 2.0, 4.0, 8.0]
order, undecided = [], []
for sensitivity in sensitivities:
    trajectory = simulate(ThreeStateOpinionSystem(sensitivity=sensitivity), until=100, random_state=0)
    opinions = np.array([trajectory.agents('voters', frame).values['opinion'] for frame in range(50, 101)])
    order.append(np.mean(np.abs(opinions.mean(axis=1))))
    undecided.append(np.mean(opinions == 0))
grid = np.linspace(0.3, 8, 100)
mean_field = []
for beta in grid:
    x = 0.5
    for _ in range(500):
        x = np.tanh(beta * 4 * x / 2) / 2
    mean_field.append(abs(x))
fig, (ordering, shares) = plt.subplots(1, 2, figsize=(10, 4))
ordering.plot(sensitivities, order, 'o-', label='network')
ordering.plot(grid, mean_field, color='black', linestyle=':', label='mean field')
ordering.set_xlabel('sensitivity beta')
ordering.set_ylabel('|mean opinion|')
ordering.set_title('Order sets in with the sensitivity')
ordering.legend()
shares.plot(sensitivities, undecided, 'o-')
shares.axhline(0.5, color='black', linestyle=':')
shares.set_ylim(0, 1)
shares.set_xlabel('sensitivity beta')
shares.set_ylabel('share undecided')
shares.set_title('Half the draws are undecided')
fig.tight_layout()
plt.show()
