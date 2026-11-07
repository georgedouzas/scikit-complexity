"""
# Urban scaling of interactions

A city grows by residents who link to residents close to them in hyperbolic space, and its residents also add links
among themselves at a fixed rate per resident. The links per resident therefore grow with the size of the city, and the
meetings of a city grow faster than its population, the superlinear scaling of socioeconomic quantities that Ribeiro and
Netto review near the exponent 1.15.
"""

# %%
# The city grows to 300 residents, one per step, and every resident meets each neighbour with probability one half per
# step. The view draws the residents by their meetings of the step.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import scaling
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.social.organizations import UrbanScalingInteractionsSystem

SYSTEM = 'urban_scaling_interactions'

system = UrbanScalingInteractionsSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
observables = system.get_run_settings()['observables']
trajectory = simulate(system, observables=observables, observers=[observer], random_state=0)

# %%
# Meetings against the population
# -------------------------------
# A `sweep` varies the size of the city at three rates of internal links, and `scaling(kind='allometric')` fits
# the exponent of the meetings with a confidence interval. Without internal links the exponent stays near 1, and
# it rises with the rate of the internal links. This rate per resident is the assumption of this system, and it sets the
# exponent. Papadopoulos et al. (2015) add a fixed number of internal links per step, which keeps the mean degree
# constant.

sizes = [100, 200, 400, 800]
seeds = [0, 1, 2]
rates = [0.0, 0.002, 0.005]
fig, ax = plt.subplots(figsize=(7, 4))
for rate in rates:
    bounds = {'population': sizes, 'internal': [rate], 'random_state': seeds}
    result = sweep(
        UrbanScalingInteractionsSystem(),
        bounds,
        until=float(max(sizes)),
        outputs=['total_meetings'],
        observables=observables,
        random_state=0,
    )
    meetings = result.series('total_meetings')[:, -1]
    samples = dict(zip(sizes, meetings.reshape(len(sizes), len(seeds)), strict=True))
    fit = scaling(samples, kind='allometric', resamples=500)
    label = f'internal {rate}: exponent {fit.exponent:.2f} [{fit.interval[0]:.2f}, {fit.interval[1]:.2f}]'
    ax.loglog(sizes, [np.mean(samples[size]) for size in sizes], 'o-', label=label)
ax.set_xlabel('population N')
ax.set_ylabel('meetings per step')
ax.set_title('Internal links make the meetings grow faster than the population')
ax.legend()
fig.tight_layout()
plt.show()
