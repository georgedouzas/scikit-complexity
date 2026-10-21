"""
# Cantor set

The unit interval loses its open middle third, then each of the two remaining intervals loses its middle third, and so
on, as every interval is substituted by its two outer thirds at every generation.
"""

# %%
# The interval is substituted for eight generations, so the last patch holds 256 intervals of length `3^-8`. After the
# run the first seven generations are logged once more as an image of stacked bars, the first at the top, and the
# blueprint shows them as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cantor_set'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/generations',
        name='generations 0 to 6',
        visual_bounds=rrb.VisualBounds2D(x_range=[-20, 749], y_range=[-20, 510]),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), observers=[observer]).run(8)
trajectory = simulation.trajectory_
stack = np.zeros((490, 729, 3), dtype=np.uint8)
for generation in range(7):
    for (start, _), (end, _) in trajectory.structure('cantor', generation).segments():
        stack[70 * generation : 70 * generation + 40, round(729 * start) : round(729 * end)] = [255, 170, 40]
observer.recording_.log('generations', rr.Image(stack), static=True)
observer.recording_.flush()

# %%
# Length and box-counting dimension
# ---------------------------------
# Every generation keeps two thirds of the length, so the logarithm of the total length falls on the line
# `n log(2/3)`. Boxes of side `s` along the line count the intervals at that scale, and their number should fall as
# `s^-D` with the similarity dimension `D = log 2 / log 3`, about 0.631.

generations = trajectory.times
lengths = trajectory['cantor.length']
segments = trajectory.structure('cantor', 8).segments()
sizes = 3.0 ** -np.arange(1, 7)
counts = np.array([len(np.unique(np.floor(segments[:, 0, 0] / size + 1e-9))) for size in sizes])
dimension = np.log(2) / np.log(3)
fig, (growth, boxes) = plt.subplots(1, 2, figsize=(10, 4.5))
growth.semilogy(generations, lengths, 'o', label='total length')
growth.semilogy(generations, (2 / 3) ** generations, '--', color='black', label='(2/3)^n')
growth.set_xlabel('generation n')
growth.set_ylabel('length')
growth.set_title('Length of the Cantor set')
growth.legend()
boxes.loglog(sizes, counts, 'o', label='occupied boxes')
boxes.loglog(sizes, counts[0] * (sizes / sizes[0]) ** -dimension, '--', color='black', label='slope -log 2 / log 3')
boxes.set_xlabel('box side s')
boxes.set_ylabel('occupied boxes')
boxes.set_title(f'Box counting, D = {-np.polyfit(np.log(sizes), np.log(counts), 1)[0]:.3f}')
boxes.legend()
fig.tight_layout()
plt.show()
