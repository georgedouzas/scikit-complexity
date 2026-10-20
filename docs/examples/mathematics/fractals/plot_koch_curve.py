"""
# Koch curve

A segment grows into the Koch curve as the L-system `F → F+F--F+F` replaces every segment by four segments of a third
of its length, and a turtle that turns by 60 degrees draws each word as a curve between the same two endpoints.
"""

# %%
# The word starts as one segment and is rewritten for five generations, so the last curve has 1024 segments. The
# blueprint shows the curve at the last generation, in a frame fixed across generations.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'koch_curve'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/koch',
        name='koch',
        visual_bounds=rrb.VisualBounds2D(x_range=[-0.03, 1.03], y_range=[-0.35, 0.05]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), observers=[observer]).run(5)
trajectory = simulation.trajectory_

# %%
# Length and box-counting dimension
# ---------------------------------
# Every generation multiplies the length of the curve by 4/3, so its logarithm grows on the line `n log(4/3)`. The
# number of boxes of side `s` that the raster of the last curve touches should fall as `s^-D` with the similarity
# dimension `D = log 4 / log 3`, about 1.262, over boxes larger than its segments.

generations = trajectory.times
lengths = trajectory['koch.length']
grid = trajectory.structure('koch', 5).raster(2048)
sizes = np.array([16, 32, 64, 128, 256])
counts = np.array([grid.reshape(2048 // size, size, 2048 // size, size).any(axis=(1, 3)).sum() for size in sizes])
dimension = np.log(4) / np.log(3)
fig, (growth, boxes) = plt.subplots(1, 2, figsize=(10, 4.5))
growth.semilogy(generations, lengths, 'o', label='curve length')
growth.semilogy(generations, (4 / 3) ** generations, '--', color='black', label='(4/3)^n')
growth.set_xlabel('generation n')
growth.set_ylabel('length')
growth.set_title('Length of the Koch curve')
growth.legend()
boxes.loglog(sizes, counts, 'o', label='occupied boxes')
boxes.loglog(sizes, counts[0] * (sizes / sizes[0]) ** -dimension, '--', color='black', label='slope -log 4 / log 3')
boxes.set_xlabel('box side s (pixels)')
boxes.set_ylabel('occupied boxes')
boxes.set_title(f'Box counting, D = {-np.polyfit(np.log(sizes), np.log(counts), 1)[0]:.3f}')
boxes.legend()
fig.tight_layout()
plt.show()
