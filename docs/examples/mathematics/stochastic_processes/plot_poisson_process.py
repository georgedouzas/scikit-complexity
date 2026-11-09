"""
# Poisson process

Six counters grow by one at independent, exponentially distributed waiting times: staircases with steps at random
times, each rising on average by the rate per unit time, and drifting apart as the square root of time.
"""

# %%
# The counters have rate 1 and run for 50 time units. The default blueprint shows the six counts as time series.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem

SYSTEM = 'poisson_process'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(PoissonProcessSystem(n_processes=6), dt=0.2, observers=[observer], random_state=0)

# %%
# Poisson counts
# --------------
# Over 2000 counters the count at time `t` should follow the Poisson distribution of mean `rate t`, so its Fano factor,
# the variance over the mean, should be 1 at every time, within the few percent of sampling
# error of 2000 counters, the signature of independent events.

counts = simulate(PoissonProcessSystem(n_processes=2000), random_state=1)
values = counts.get_values('counters.count')
fig, (distribution_axes, fano) = plt.subplots(1, 2, figsize=(11, 4))
grid = np.arange(25, 80)
distribution_axes.hist(values[-1], bins=np.arange(24.5, 80.5), density=True, label='simulated, t = 50')
distribution_axes.plot(grid, stats.poisson.pmf(grid, 50), 'k.-', label='Poisson, mean 50')
distribution_axes.set_xlabel('count')
distribution_axes.set_ylabel('probability')
distribution_axes.set_title('Counts at t = 50')
distribution_axes.legend()
fano.plot(counts.times[1:], distribution(values[1:].T, kind='dispersion-index').index)
fano.axhline(1, color='black', linestyle=':')
fano.set_ylim(0.5, 1.5)
fano.set_xlabel('time')
fano.set_ylabel('Fano factor')
fano.set_title('Variance equals the mean')
fig.tight_layout()
plt.show()
