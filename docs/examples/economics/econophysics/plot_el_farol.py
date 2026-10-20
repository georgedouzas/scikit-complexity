"""
# El Farol bar

A hundred people decide every week whether to go to the El Farol bar, which is enjoyable only when at most 60 go.
Nobody can deduce the right choice, since a forecast that everyone believes defeats itself, so each person acts on
the most accurate of a few rules of thumb. The attendance never settles, yet its mean organizes itself near the
capacity of 60.
"""

# %%
# Every patron holds six predictors of next week's attendance, drawn from a pool of 46: cycle detectors, moving
# averages, trends, mirror images, and anchors on fixed levels. It goes when its currently most accurate predictor
# forecasts fewer than 60 patrons. The bar counts the patrons and reports the attendance, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'el_farol'
WEEKS, TRANSIENT, CAPACITY = 500, 100, 60

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters/bar.attendance',
        name='attendance',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, n_agents=100, capacity=CAPACITY, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=WEEKS)

# %%
# Self-organization around the capacity
# -------------------------------------
# Arthur found that the mean attendance converges to the capacity, while the attendance keeps fluctuating, since any
# predictor that becomes popular defeats itself. The running mean after the transient should approach 60.

times = simulation.trajectory_.times
attendance = simulation.trajectory_['bar.attendance']
settled = attendance[TRANSIENT:]
running = np.cumsum(settled) / np.arange(1, len(settled) + 1)
fig, (series, histogram) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(times, attendance, lw=0.6, label='attendance')
series.plot(times[TRANSIENT:], running, color='black', label='running mean after the transient')
series.axhline(CAPACITY, color='C3', linestyle='--', label='capacity')
series.set_xlabel('week')
series.set_ylabel('patrons')
series.set_title(f'Mean attendance {settled.mean():.1f}')
series.legend()
histogram.hist(settled, bins=20, density=True)
histogram.axvline(CAPACITY, color='C3', linestyle='--')
histogram.set_xlabel('attendance')
histogram.set_ylabel('density')
histogram.set_title('Distribution of the attendance')
fig.tight_layout()
plt.show()
