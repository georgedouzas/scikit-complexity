"""
# Ehrenfest urn

Forty numbered balls start in the first of two boxes, and at every step a ball drawn at random moves to the other box.
The count of the first box falls quickly toward twenty, half the balls, and then fluctuates around it without ever
returning near forty, although every step can be undone. The view shows the count of the first box over 250 steps.
"""

# %%
# The model is created by name from the catalog with 40 balls, all in the first box. One step moves one ball, and the
# blueprint shows the count over the whole run.

import math

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ehrenfest_urn'
BALLS = 40

# The view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/urn/n', name='balls in the first box', axis_x=whole_run),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_balls=BALLS), dt=1, observers=[observer], random_state=0).run(250)

# %%
# Relaxation and the binomial distribution
# ----------------------------------------
# The mean count relaxes as `N / 2 + (N / 2) (1 - 2 / N)^t`, so the mean over 200 urns should follow that curve. Once
# relaxed, the count of one long run should be distributed as the binomial distribution of `N` balls with
# probability one half, of mean `N / 2` and variance `N / 4`.

steps = np.arange(301)
urns = np.array(
    [
        Simulation(create(MODEL, n_balls=BALLS), dt=1, random_state=seed).run(300).trajectory_['urn.n']
        for seed in range(200)
    ],
)
counts = np.asarray(Simulation(create(MODEL, n_balls=BALLS), dt=1, random_state=1).run(20000).trajectory_['urn.n'])
relaxed = counts[400:].astype(int)
values = np.arange(BALLS + 1)

fig, (relaxation, distribution) = plt.subplots(1, 2, figsize=(10, 4))
relaxation.plot(steps, urns.mean(axis=0), label='mean of 200 urns')
relaxation.plot(steps, BALLS / 2 + BALLS / 2 * (1 - 2 / BALLS) ** steps, 'k--', label='N/2 + (N/2)(1 - 2/N)^t')
relaxation.set_xlabel('step')
relaxation.set_ylabel('balls in the first box')
relaxation.set_title('Exponential relaxation of the mean')
relaxation.legend()
distribution.bar(values, np.bincount(relaxed, minlength=BALLS + 1) / relaxed.size, label='one run, after relaxation')
distribution.plot(values, [math.comb(BALLS, k) / 2**BALLS for k in values], 'ko-', markersize=3, label='binomial')
distribution.set_xlabel('balls in the first box')
distribution.set_ylabel('frequency')
distribution.set_title(f'Mean {relaxed.mean():.1f}, variance {relaxed.var():.1f}')
distribution.legend()
fig.tight_layout()
plt.show()
