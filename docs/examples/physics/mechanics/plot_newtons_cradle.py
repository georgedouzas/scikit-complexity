"""
# Newton's cradle

The first ball of the cradle swings into the row, and the last one flies out with the same speed while the others stay
almost still: elastic collisions between equal masses pass the whole momentum and energy down the row.
"""

# %%
# Five balls of diameter 0.1 hang with gaps of 0.001, and the first starts towards the others at speed 0.5. The default
# blueprint shows the positions and velocities of the first three balls, where the velocity jumps from one ball to the
# next at every collision.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'newtons_cradle'

system = create(MODEL, n_balls=5, speed=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=5)

# %%
# Momentum passed down the row
# ----------------------------
# Equal masses exchange velocities at every collision, so the velocity of the first ball should pass to the last
# within an instant, keeping the speed 0.5, and come back to the first ball half a swing later, while the middle balls
# stay almost at rest.

times = np.asarray(simulation.trajectory_.times)
velocities = np.stack([np.ravel(simulation.trajectory_[f'ball_{index}.x_d1']) for index in range(5)], axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
for index in range(5):
    ax.plot(times, velocities[:, index], label=f'ball {index}')
ax.set_xlabel('time')
ax.set_ylabel('velocity')
ax.set_title('The velocity jumps from the first ball to the last')
ax.legend(ncol=5, loc='lower center')
plt.show()
