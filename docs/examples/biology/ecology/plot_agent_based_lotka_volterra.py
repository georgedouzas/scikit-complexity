"""
# Agent-based Lotka-Volterra

Prey and predators walk on a grid: prey breed where they are not crowded, predators eat the prey they meet and breed
after a meal. Their numbers cycle out of phase like the Lotka-Volterra equations, but demographic noise makes the
cycles irregular.
"""

# %%
# The model is created by name from the catalog with 300 prey and 60 predators on a 30 by 30 grid. The observer saves
# every step with the default blueprint, whose primary view shows the animals on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'agent_based_lotka_volterra'

system = create(MODEL, prey=300, predators=60, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# Cycles out of phase
# -------------------
# The predators should peak after the prey, so the cross-correlation of the two counts should be largest at a
# positive lag of the predators, and the orbit in the phase plane should circle counterclockwise.

trajectory = simulation.trajectory_
prey, predators = trajectory['prey.count'], trajectory['predators.count']
lags = np.arange(0, 40)
correlation = [np.corrcoef(prey[: len(prey) - lag], predators[lag:])[0, 1] for lag in lags]
fig, (orbit, cross) = plt.subplots(1, 2, figsize=(11, 4))
orbit.plot(prey, predators, linewidth=0.8)
orbit.plot(prey[0], predators[0], 'o', color='black', label='start')
orbit.set_xlabel('prey')
orbit.set_ylabel('predators')
orbit.set_title('Noisy predator-prey cycles')
orbit.legend()
cross.plot(lags, correlation, 'o-')
cross.set_xlabel('lag of the predators (steps)')
cross.set_ylabel('correlation with the prey')
cross.set_title(f'Predators lag the prey by {lags[np.argmax(correlation)]} steps')
fig.tight_layout()
plt.show()
