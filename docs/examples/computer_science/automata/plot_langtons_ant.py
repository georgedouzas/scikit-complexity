"""
# Langton's ant

The ant turns right on white cells and left on black ones, and flips the colour of every cell it leaves. For about
10,000 steps it builds an irregular blob, then it suddenly settles into a cycle of 104 steps that moves it two cells
along a diagonal: the highway, which it builds forever.
"""

# %%
# The torus has 120 by 120 cells, large enough for the highway to grow away from the blob for the 12,000 steps of the
# run, and the observer saves a frame every 100 steps. The image shows the black cells and the ant.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'langtons_ant'
CENTER = 60

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, shape=(120, 120)), dt=100, observers=[observer]).run(until=12000)

# %%
# From the blob to the highway
# ----------------------------
# During the chaotic phase the ant should stay within a few tens of cells of its start, like a confined random walk.
# Once on the highway its distance should grow linearly, by `2√2` cells every 104 steps, and the number of black cells
# by 12 every 104 steps.

trajectory = simulation.trajectory_
colors, ant = trajectory.lattice('cells.color'), trajectory.lattice('cells.ant')
positions = np.array([np.argwhere(frame >= 0)[0] for frame in ant])
distance = np.hypot(*(positions - CENTER).T)
steps = trajectory.times
fig, (away, black) = plt.subplots(1, 2, figsize=(11, 4))
away.plot(steps, distance, label='simulated')
highway = steps >= 10400
slope = 2 * np.sqrt(2) / 104
away.plot(
    steps[highway],
    distance[highway][0] + slope * (steps[highway] - steps[highway][0]),
    'k--',
    label='2√2 per 104',
)
away.set_xlabel('step')
away.set_ylabel('distance from the start')
away.set_title('Confined, then escaping on the highway')
away.legend()
counts = colors.sum(axis=(1, 2))
black.plot(steps, counts, label='simulated')
black.plot(
    steps[highway],
    counts[highway][0] + 12 / 104 * (steps[highway] - steps[highway][0]),
    'k--',
    label='12 per 104',
)
black.set_xlabel('step')
black.set_ylabel('black cells')
black.set_title('The highway adds black cells at a constant rate')
black.legend()
fig.tight_layout()
plt.show()
