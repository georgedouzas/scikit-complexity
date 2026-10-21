"""
# LEGION

A lattice of relaxation oscillators segments a stimulus into its objects. Oscillators of one connected object excite
each other and fire together, and a global inhibitor lets only one object be active at a time, so the two objects of
the stimulus take turns, each one a synchronized block.
"""

# %%
# The model is created by name from the catalog with its default stimulus of two objects on a 4 by 4 lattice, an
# L-shape at the top left and a bar at the right, and the faster slow time scale `epsilon = 0.1`, so that several
# turns fit in the run. The observer saves every step with the default blueprint, whose primary view shows the
# oscillators.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'legion'

system = create(MODEL, epsilon=0.1, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=290)

# %%
# Synchrony within, alternation between
# -------------------------------------
# An oscillator is active while its fast variable is positive. The three sites of the first object should be active
# at the same times, the two sites of the second object at the same times, the two objects never together, and the
# unstimulated sites never.

trajectory = simulation.trajectory_
active = fields_of(trajectory, 'oscillators.x') > 0
sites = {'object 1: (0, 0)': (0, 0), 'object 1: (0, 1)': (0, 1), 'object 1: (1, 0)': (1, 0)}
sites |= {'object 2: (2, 3)': (2, 3), 'object 2: (3, 3)': (3, 3), 'unstimulated: (1, 1)': (1, 1)}
fig, ax = plt.subplots(figsize=(9, 3.5))
for row, (i, j) in enumerate(sites.values()):
    ax.fill_between(trajectory.times, row - 0.4, row + 0.4, where=active[:, i, j], step='mid', color=f'C{row // 3}')
ax.set_yticks(range(len(sites)), list(sites))
ax.set_xlabel('time')
overlap = np.sum(active[:, 0, 0] & active[:, 3, 3])
ax.set_title(f'Objects fire in turns; steps with both objects active: {overlap}')
fig.tight_layout()
plt.show()
