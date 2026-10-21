"""
# Random fragmentation

An object of a million units breaks at a uniformly random point, and so do the pieces, each with probability 1/2 per
step. The number of fragments grows geometrically while the total size stays fixed, and the logarithms of the sizes
spread into a bell, as a sum of many independent random factors.
"""

# %%
# The object has size `10^6` and breaks for 14 steps, before many fragments reach the smallest size 1. After the run,
# the fragments of every step are logged as one row of an image, sorted from the largest to the smallest and laid
# end to end in alternating colours, the first step at the top, and the blueprint puts it first, as the primary view.

import itertools

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fragmentation'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/sizes',
        name='fragments by step, sorted by size',
        visual_bounds=rrb.VisualBounds2D(x_range=[0, 800], y_range=[0, 225]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, size=10**6), dt=1, observers=[observer], random_state=0).run(until=14)
trajectory = simulation.trajectory_
lengths = [trajectory.agents('fragments', frame).values['length'] for frame in range(len(trajectory.times))]
rows = np.zeros((15 * len(lengths), 800, 3), dtype=np.uint8)
for step, row in enumerate(lengths):
    edges = np.round(800 * np.concatenate([[0], np.cumsum(np.sort(row)[::-1])]) / 10**6).astype(int)
    for index, (left, right) in enumerate(itertools.pairwise(edges)):
        color = [255, 170, 40] if index % 2 == 0 else [90, 170, 255]
        rows[15 * step + 2 : 15 * step + 13, left:right] = color
observer.recording_.log('sizes', rr.Image(rows), static=True)
observer.recording_.flush()

# %%
# Geometric growth and log-normal sizes
# -------------------------------------
# Every fragment breaks with probability 1/2 per step, so the number of fragments should grow by the factor 1.5 per
# step once there are many of them. A uniform cut multiplies a size by `U` with `E[ln U] = -1` and `Var[ln U] = 1`,
# so the mean of `ln L` should fall by 0.5 per step, its variance should grow by about 0.75 per step, and the total
# size should stay `10^6`.

steps = trajectory.times
logs = [np.log(row) for row in lengths]
fig, (number, spread) = plt.subplots(1, 2, figsize=(11, 4))
number.semilogy(steps, [len(row) for row in lengths], 'o', label='fragments')
number.semilogy(steps, len(lengths[-1]) * 1.5 ** (steps - steps[-1]), 'k--', label='slope of $1.5^t$')
number.set_xlabel('step')
number.set_ylabel('number of fragments')
number.set_title(f'Geometric growth; total size {lengths[-1].sum():.0f}')
number.legend()
spread.plot(steps, [row.mean() for row in logs], 'o', label='mean of ln L')
spread.plot(steps, np.log(1e6) - 0.5 * steps, 'k--', label='ln 10^6 - 0.5 t')
spread.plot(steps, [row.var() for row in logs], 's', label='variance of ln L')
spread.plot(steps, 0.75 * steps, 'k:', label='0.75 t')
spread.set_xlabel('step')
spread.set_ylabel('log size statistics')
spread.set_title('Log sizes drift and spread linearly')
spread.legend(fontsize=8)
fig.tight_layout()
plt.show()
