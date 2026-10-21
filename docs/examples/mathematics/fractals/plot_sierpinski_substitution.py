"""
# Sierpiński substitutions

A triangle becomes the Sierpiński gasket as every triangle is replaced by its three corner copies of half its side,
leaving holes at every scale. The same substitution of a square by the eight copies of a third of its side around its
middle gives the Sierpiński carpet, shown in the figure below.
"""

# %%
# The triangle is substituted for six generations, so the last patch holds 729 triangles, drawn as a filled image.
# The blueprint shows the gasket at the last generation.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sierpinski_substitution'

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(origin='/sierpinski', name='sierpinski'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), observers=[observer]).run(6)
trajectory = simulation.trajectory_

# %%
# Area and box-counting dimension
# -------------------------------
# Every generation keeps three quarters of the area of the gasket and eight ninths of the area of the carpet, so the
# logarithms of the areas fall on straight lines. The number of boxes of side `s` that the raster of the gasket
# touches should fall as `s^-D` with the similarity dimension `D = log 3 / log 2`, about 1.585, and the carpet's with
# `D = log 8 / log 3`, about 1.893.

carpet = Simulation(create(MODEL, shape='carpet')).run(4).trajectory_
generations = trajectory.times
gasket_area = trajectory['sierpinski.area'] / trajectory['sierpinski.area'][0]
carpet_area = carpet['sierpinski.area']
sizes = np.array([4, 8, 16, 32, 64, 128])
fig, (decay, boxes) = plt.subplots(1, 2, figsize=(10, 4.5))
decay.semilogy(generations, gasket_area, 'o', label='gasket')
decay.semilogy(generations, 0.75**generations, '--', color='black', label='(3/4)^n')
decay.semilogy(carpet.times, carpet_area, 's', label='carpet')
decay.semilogy(carpet.times, (8 / 9) ** carpet.times, ':', color='black', label='(8/9)^n')
decay.set_xlabel('generation n')
decay.set_ylabel('area relative to generation 0')
decay.set_title('Area of the substitutions')
decay.legend()
for name, structure, dimension in [
    ('gasket', trajectory.structure('sierpinski', 6), np.log(3) / np.log(2)),
    ('carpet', carpet.structure('sierpinski', 4), np.log(8) / np.log(3)),
]:
    grid = structure.raster(512)
    counts = np.array([grid.reshape(512 // size, size, 512 // size, size).any(axis=(1, 3)).sum() for size in sizes])
    boxes.loglog(sizes, counts, 'o', label=f'{name}, slope -{dimension:.3f}')
    boxes.loglog(sizes, counts[0] * (sizes / sizes[0]) ** -dimension, '--', color='black')
boxes.set_xlabel('box side s (pixels)')
boxes.set_ylabel('occupied boxes')
boxes.set_title('Box counting')
boxes.legend()
fig.tight_layout()
plt.show()
