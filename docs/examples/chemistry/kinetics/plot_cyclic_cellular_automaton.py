"""
# Cyclic cellular automaton

Every cell advances to the next of 14 levels when a neighbour already holds it. From random levels, small patches
first lock into local cycles, then spiral waves form around their defects and grow until they tile the whole grid.
"""

# %%
# The system is built from its class on a 128 by 128 grid with 14 levels and threshold 1, from random
# levels, and runs for 300 steps. The default blueprint shows the levels on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import correlation
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import CyclicCellularAutomatonSystem

SYSTEM = 'cyclic_cellular_automaton'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(CyclicCellularAutomatonSystem((128, 128), random_state=0), until=300, observers=[observer])

# %%
# Self-organization into spirals
# ------------------------------
# A cell advances when it changes level. In the random initial state few cells have a neighbour at the next level, and
# where cycles lock in every cell advances once per step until it is part of a wave, so the share of advancing cells
# should rise to a plateau as the spirals take over. The waves also correlate the levels over growing distances, so
# the spatial correlation function should decay more slowly at later steps.

levels = trajectory.get_lattice('cells.level')
advancing = np.mean(levels[1:] != levels[:-1], axis=(1, 2))
fig, (activity, reach) = plt.subplots(1, 2, figsize=(10, 4))
activity.plot(trajectory.times[1:], advancing)
activity.set_xlabel('step')
activity.set_ylabel('share of cells that advance')
activity.set_title('Activity rises as the spirals take over')
for step in (0, 20, 60, 300):
    phase = correlation(np.cos(2 * np.pi * levels[step] / 14), kind='function', max_distance=40)
    reach.plot(phase.distances, phase.correlation, label=f'step {step}')
reach.axhline(0.0, color='gray', linewidth=0.8)
reach.set_xlabel('distance (cells)')
reach.set_ylabel('correlation of the phase')
reach.set_title('Correlations reach farther over time')
reach.legend()
fig.tight_layout()
plt.show()
