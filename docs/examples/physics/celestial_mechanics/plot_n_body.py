"""
# N-body

Three equal stars chase each other along one figure-eight curve, the periodic orbit of Chenciner and Montgomery: each
star crosses the middle in turn, and after one period every star is back where it started.
"""

# %%
# The preset `'figure_eight'` gives the masses, positions, and velocities of the orbit, with period 6.3259 for
# `G = 1`. The run covers one period, and the default blueprint shows the stars in the plane with their paths.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'n_body'
PERIOD = 6.3259

system = create(MODEL, preset='figure_eight')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=PERIOD)

# %%
# Return after one period
# -----------------------
# Over three periods, the distance of the configuration from its initial state should fall back to almost zero at
# every multiple of the period, and the three stars should share a single figure-eight path.

run = Simulation(create(MODEL, preset='figure_eight'), dt=0.005).run(until=3 * PERIOD)
positions = np.stack([np.asarray(run.trajectory_[f'body_{index}.x']) for index in range(3)], axis=1)
distance = np.linalg.norm(positions - positions[0], axis=(1, 2))
fig, (paths, returns) = plt.subplots(1, 2, figsize=(10, 4))
for index in range(3):
    paths.plot(positions[:, index, 0], positions[:, index, 1], linewidth=1, label=f'star {index}')
paths.set_aspect('equal')
paths.set_xlabel('x')
paths.set_ylabel('y')
paths.set_title('One shared figure-eight path')
paths.legend()
returns.plot(run.trajectory_.times, distance)
for multiple in range(1, 4):
    returns.axvline(multiple * PERIOD, linestyle='--', color='grey')
returns.set_xlabel('time')
returns.set_ylabel('distance from the initial configuration')
returns.set_title('Periodic return, dashed lines at multiples of 6.3259')
fig.tight_layout()
plt.show()
