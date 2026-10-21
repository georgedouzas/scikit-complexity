"""
# Continuous-time random walk

Six walkers on the integers step left or right with equal probability at exponentially distributed times. Their
paths spread without a preferred direction, and their typical distance from the origin grows only as the square root
of time: diffusion.
"""

# %%
# The walkers step at rate 1 without bias for 100 time units. The default blueprint shows their positions as time
# series.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'random_walk'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, n_walkers=6), dt=0.5, observers=[observer], random_state=0).run(until=100)

# %%
# Diffusive spreading
# -------------------
# Each step moves a walker by ±1 and steps come at rate 1, so over 2000 walkers the mean position should stay at 0 and
# the variance should grow as `t`, with or without bias, since steps come as a Poisson stream of unit jumps. The
# positions at `t = 100` should follow the Gaussian of variance 100, and with a bias of 0.7 to the right the mean
# should drift at `2 * 0.7 - 1 = 0.4` per unit time.

fig, (spread, distribution) = plt.subplots(1, 2, figsize=(11, 4))
for bias in (0.5, 0.7):
    run = Simulation(create(MODEL, n_walkers=2000, bias=bias), dt=1.0, random_state=1).run(until=100).trajectory_
    positions = run['walkers.x']
    spread.plot(run.times, positions.var(axis=1), label=f'variance, bias {bias}')
    spread.plot(run.times, positions.mean(axis=1), '--', label=f'mean, bias {bias}')
    if bias == 0.5:
        final = positions[-1]
spread.plot([0, 100], [0, 100], 'k:', label='t')
spread.plot([0, 100], [0, 40], 'k-.', label='0.4 t')
spread.set_xlabel('time')
spread.set_ylabel('position statistics')
spread.set_title('Variance grows as t')
spread.legend(fontsize=8)
grid = np.arange(-40, 41)
distribution.hist(final, bins=np.arange(-40.5, 41.5, 2), density=True, label='simulated, unbiased, t = 100')
distribution.plot(grid, stats.norm.pdf(grid, 0, 10), 'k', label='Gaussian, variance 100')
distribution.set_xlabel('position')
distribution.set_ylabel('probability')
distribution.set_title('Positions are Gaussian')
distribution.legend(fontsize=8)
fig.tight_layout()
plt.show()
