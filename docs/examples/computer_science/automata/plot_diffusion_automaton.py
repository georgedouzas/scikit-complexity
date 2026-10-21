"""
# Diffusion automaton

Every cell passes a fixed share of its difference with each of its four neighbours, so a square of ones loses its
corners, rounds off, and spreads into a Gaussian bump whose width grows as the square root of time, while the total
content stays exactly the same.
"""

# %%
# The torus has 80 by 80 cells and starts with a 10 by 10 square of ones in its middle, and the exchange share is
# `r = 0.1` for 200 steps. The image shows the content of every cell.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'diffusion_automaton'
R = 0.1

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, shape=(80, 80), r=R, width=10), dt=1, observers=[observer]).run(until=200)

# %%
# Spreading and conservation
# --------------------------
# On the lattice each step adds `2r` to the variance of the content along each axis, so the variance should grow as
# `(10² - 1) / 12 + 2 r t` from the square's start, the profile through the centre should approach a Gaussian of that
# variance, and the total content should stay 100 to rounding error.

levels = simulation.trajectory_.lattice('cells.level')
times = simulation.trajectory_.times
coordinates = np.arange(80) - 39.5
total = levels.sum(axis=(1, 2))
variance = (levels.sum(axis=2) * coordinates**2).sum(axis=1) / total
fig, (spread, profile) = plt.subplots(1, 2, figsize=(11, 4))
spread.plot(times, variance, label='simulated')
spread.plot(times, 99 / 12 + 2 * R * times, 'k--', label='$99/12 + 2rt$')
spread.set_xlabel('step')
spread.set_ylabel('variance along x')
spread.set_title(f'Linear spreading; total drift {np.ptp(total):.1e}')
spread.legend()
final = 99 / 12 + 2 * R * times[-1]
profile.plot(coordinates, levels[-1, 40], 'o', markersize=3, label='simulated, step 200')
profile.plot(coordinates, 100 / (2 * np.pi * final) * np.exp(-(coordinates**2) / (2 * final)), 'k', label='Gaussian')
profile.set_xlabel('cell')
profile.set_ylabel('content')
profile.set_title('The square becomes a Gaussian')
profile.legend()
fig.tight_layout()
plt.show()
