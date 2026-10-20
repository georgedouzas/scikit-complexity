"""
# Mandelbrot set

Every point `c` of the plane iterates `z -> z^2 + c` from zero. The points of the Mandelbrot set never leave the disk
of radius 2, so their escape count keeps growing, while every other point escapes after a few steps and its count
stops. The escape times draw bands that wrap ever closer around the set as the run goes on.
"""

# %%
# The model is created by name from the catalog on a lattice of 240 by 300 points that covers the set. The map runs
# for 100 steps, and the observer saves every fifth frame with the default blueprint, whose primary view is the image
# of the escape counts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'mandelbrot'
WINDOW = (-2.2, 0.8, -1.2, 1.2)

system = create(MODEL, shape=(240, 300), window=WINDOW)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=100)

# %%
# Escape times and the area of the set
# ------------------------------------
# The points still bounded after `k` steps cover a region that shrinks towards the set as `k` grows. Its area should
# fall quickly at first and then level off near the area of the Mandelbrot set, about 1.506.

counts = simulation.trajectory_.lattice('plane.n')
cell = (WINDOW[1] - WINDOW[0]) / (counts.shape[2] - 1) * (WINDOW[3] - WINDOW[2]) / (counts.shape[1] - 1)
steps = simulation.trajectory_.times
bounded = (counts == steps[:, np.newaxis, np.newaxis]).sum(axis=(1, 2)) * cell
fig, (image, area) = plt.subplots(1, 2, figsize=(11, 4))
image.imshow(np.log1p(counts[-1]), extent=WINDOW, cmap='magma')
image.set_xlabel('real part of c')
image.set_ylabel('imaginary part of c')
image.set_title('Escape counts after 100 steps (log scale)')
area.plot(steps[1:], bounded[1:])
area.axhline(1.506, color='black', linestyle='--', label='area of the set')
area.set_xscale('log')
area.set_xlabel('step')
area.set_ylabel('area of the bounded points')
area.set_title('The bounded region shrinks to the set')
area.legend()
fig.tight_layout()
plt.show()
