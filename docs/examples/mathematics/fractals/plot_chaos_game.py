"""
# Chaos game

Twenty thousand points scattered over the unit square fall onto the Sierpiński triangle within a few steps, as each
point moves at every step half way towards one of the three corners, chosen at random.
"""

# %%
# The points start uniformly in the unit square and iterate the three maps of the Sierpiński triangle with equal
# weights for 30 steps. The blueprint shows the cloud of points at the last frame, upright.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import box_counting_dimension
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chaos_game'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(origin='/upright', name='points at the last frame'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, preset='sierpinski', n_points=20000)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(30)
# Rerun's 2D views point y downward, so the last cloud is logged again with y flipped to stand the triangle upright.
last = simulation.trajectory_['points.p'][-1]
observer.recording_.log('upright', rr.Points2D(last * [1, -1], radii=0.002, colors=[200, 90, 220]), static=True)
observer.recording_.flush()

# %%
# Box-counting dimension and the Barnsley fern
# --------------------------------------------
# The number of boxes of side `s` that the points of the triangle occupy should grow as `s^-D` with the box-counting
# dimension `D = log 3 / log 2`, about 1.585, on a grid of 128 by 128 cells. The same game with the four maps of
# Barnsley, drawn with the probabilities 0.01, 0.85, 0.07, and 0.07, draws a fern.

points = simulation.trajectory_['points.p'][-1]
cells = np.clip((points * 128).astype(int), 0, 127)
grid = np.zeros((128, 128), dtype=bool)
grid[cells[:, 1], cells[:, 0]] = True
sizes = np.array([1, 2, 4, 8, 16, 32])
counts = [grid.reshape(128 // size, size, 128 // size, size).any(axis=(1, 3)).sum() for size in sizes]
dimension = box_counting_dimension(grid, sizes)
fern = Simulation(create(MODEL, preset='barnsley', n_points=20000), dt=30, random_state=0).run(30)
leaves = fern.trajectory_['points.p'][-1]
fig, (boxes, plant) = plt.subplots(1, 2, figsize=(10, 4.5))
boxes.loglog(sizes, counts, 'o', label='occupied boxes')
boxes.loglog(sizes, counts[0] * sizes ** -(np.log(3) / np.log(2)), '--', color='black', label='slope -log 3 / log 2')
boxes.set_xlabel('box side s (cells)')
boxes.set_ylabel('occupied boxes')
boxes.set_title(f'Sierpiński triangle, D = {dimension:.3f}')
boxes.legend()
plant.scatter(leaves[:, 0], leaves[:, 1], s=0.1, color='green')
plant.set_aspect('equal')
plant.set_xlabel('x')
plant.set_ylabel('y')
plant.set_title('Barnsley fern')
fig.tight_layout()
plt.show()
