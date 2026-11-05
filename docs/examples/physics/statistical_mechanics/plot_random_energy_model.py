"""
# Random energy model

The `2^N` energies of the random energy model are independent Gaussians of variance `N / 2`. Its Boltzmann sum is exact:
the entropy per spin follows `ln 2 - 1 / (4 T^2)` above the glass temperature `T_g = 1 / (2 sqrt(ln 2))`, and below it
the weight freezes on the lowest levels, so the entropy falls to zero.
"""

# %%
# The model sums over 4096 levels in one static frame, with the log partition function reduced in log space.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create
from skcomplex.models.physics.statistical_mechanics import random_energy_model

MODEL = 'random_energy_model'

system = create(MODEL, n_spins=12, temperature=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer]).prepare().run(until=1.0)

# %%
# Entropy against temperature
# ---------------------------
# The energies are quenched, so a disorder average sweeps the `random_state` of the factory as an axis of the design. A
# finite system keeps some entropy below `T_g`, and deviates from the line just above it.

temperatures = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0])
seeds = list(range(5))
bounds = {'n_spins': [16], 'temperature': temperatures.tolist(), 'random_state': seeds}
experiment = Experiment(random_energy_model, GridDesign(), bounds, until=1.0, outputs=['ensemble.entropy'])
entropies = experiment.prepare().run().result_.series('ensemble.entropy')[:, 0]
grid = np.linspace(1 / (2 * np.sqrt(np.log(2))), 2.1, 100)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temperatures, entropies.reshape(len(temperatures), len(seeds)).mean(axis=1), 'o', label='N = 16, 5 draws')
ax.plot(grid, np.log(2) - 1 / (4 * grid**2), color='black', label='ln 2 - 1 / (4 T^2)')
ax.axvline(grid[0], color='gray', linestyle='--', label='T_g')
ax.set_xlabel('temperature T')
ax.set_ylabel('entropy per spin')
ax.set_title('The entropy freezes at the glass temperature')
ax.legend()
fig.tight_layout()
plt.show()
