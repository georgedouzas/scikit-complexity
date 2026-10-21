"""
# Magnetic pendulum

A magnetic bob swinging over three magnets wanders between them in a tangled path before friction makes it settle over
one. Which magnet wins depends so sensitively on where the bob is released that the basins of the magnets are fractal.
"""

# %%
# The bob is released at rest at `(1, 1.5)` above three magnets on the unit circle, with the restoring frequency 1 and
# the friction `q = 0.2`. The default blueprint shows the bob in the plane with its path, which spirals among the
# magnets before coming to rest.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'magnetic_pendulum'

system = create(MODEL, x=1.0, y=1.5, n_magnets=3)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=25)

# %%
# Sensitive dependence on the release point
# -----------------------------------------
# Along a line of 80 release points, the magnet nearest to the final resting place should switch irregularly among all
# three, with short runs of one magnet broken by isolated releases that end at another, the mark of intertwined,
# fractal basins. Only the outer ends of the line, far from the magnets, give long runs of the same magnet.

angles = 2 * np.pi * np.arange(3) / 3
magnets = np.column_stack([np.cos(angles), np.sin(angles)])
starts = np.linspace(-2.0, 2.0, 80)
winners = []
for start in starts:
    run = Simulation(create(MODEL, x=start, y=1.5, n_magnets=3), dt=0.1).run(until=40)
    final = np.asarray(run.trajectory_['bob.x'])[-1]
    winners.append(np.argmin(np.linalg.norm(magnets - final, axis=1)))
fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(starts, winners, c=winners, cmap='viridis', s=18)
ax.set_yticks(range(3))
ax.set_xlabel('release x (with y = 1.5)')
ax.set_ylabel('magnet where the bob rests')
ax.set_title('Final magnet against the release point')
fig.tight_layout()
plt.show()
