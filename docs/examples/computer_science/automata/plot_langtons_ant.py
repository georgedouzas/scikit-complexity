"""
# Langton's ant

The ant turns right on white cells and left on black ones, and flips the colour of every cell it leaves. For about
10,000 steps it builds an irregular blob, then it suddenly settles into a cycle of 104 steps that moves it two cells
along a diagonal: the highway, which it builds forever.
"""

# %%
# The torus has 120 by 120 cells, large enough for the highway to grow away from the blob for the 12,000 steps of the
# run, and the observer saves a frame every 100 steps. The run records the cell and the heading of the ant and the
# colours of the torus at every frame.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import LangtonsAntSystem

SYSTEM = 'langtons_ant'
CENTER = 60

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = LangtonsAntSystem(shape=(120, 120)).prepare().simulate(until=12000, observers=[observer])

# %%
# From the blob to the highway
# ----------------------------
# During the chaotic phase the ant should stay within a few tens of cells of its start, like a confined random walk.
# Once on the highway its distance should grow linearly, by `2√2` cells every 104 steps, and the number of black cells
# by 12 every 104 steps.

rows, columns = trajectory.get_values('ant.row').ravel(), trajectory.get_values('ant.column').ravel()
distance = np.hypot(rows - CENTER, columns - CENTER)
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
counts = np.array([colors.sum() for colors in trajectory.get_values('ant.colors')[:, 0]])
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

# %%
# The pattern at the end
# ----------------------
# The last frame should show the irregular blob around the start and the highway leaving it along a diagonal, with the
# ant at its tip.

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(trajectory.get_values('ant.colors')[-1, 0], cmap='binary')
ax.plot(columns[-1], rows[-1], 'r.', markersize=10)
ax.set_title('Black cells and the ant after 12,000 steps')
ax.set_axis_off()
plt.show()
