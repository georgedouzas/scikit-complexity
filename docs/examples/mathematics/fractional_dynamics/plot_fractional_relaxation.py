"""
# Fractional relaxation

A state whose Caputo derivative of order `alpha` relaxes it, `D^alpha x = -x`, decays as the Mittag-Leffler function
`E_alpha(-t^alpha)`: faster than the exponential at first, then far slower, as the power law `t^(-alpha)`. The lower
the order, the longer the memory of the start and the heavier the tail.
"""

# %%
# The model runs with order 0.6 and logs `x` as a time series. Runs of orders 1, 0.8, 0.6, and 0.4 from the same state
# are logged once as curves of `x` against time, and the blueprint puts that view first, as the primary view.

import math

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import mittag_leffler
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fractional_relaxation'
ORDERS = (1.0, 0.8, 0.6, 0.4)
COLORS = [[90, 170, 255], [120, 210, 120], [255, 170, 40], [235, 90, 90]]

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/orders',
            name='x(t) for orders 1, 0.8, 0.6, 0.4',
            visual_bounds=rrb.VisualBounds2D(x_range=[-10, 210], y_range=[-105, 5]),
        ),
        rrb.TimeSeriesView(origin='/relaxation', name='x, order 0.6', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, order=0.6), dt=0.1, observers=[observer]).run(until=20)
runs = {order: Simulation(create(MODEL, order=order), dt=0.1).run(until=20).trajectory_ for order in ORDERS}
strips = [np.column_stack([10 * run.times, -100 * run['relaxation.x']]) for run in runs.values()]
observer.recording_.log('orders', rr.LineStrips2D(strips, radii=0.8, colors=COLORS), static=True)
observer.recording_.flush()

# %%
# Mittag-Leffler decay
# --------------------
# Every simulated curve should lie on its Mittag-Leffler function `E_alpha(-t^alpha)`, and at long times on the power
# law `t^(-alpha) / Gamma(1 - alpha)`, a straight line on logarithmic axes, while order 1 drops as `exp(-t)`.

fig, ax = plt.subplots(figsize=(6, 4.5))
for (order, run), color in zip(runs.items(), COLORS, strict=True):
    times = run.times[1:]
    exact = mittag_leffler(order, 1.0, -(times**order))
    ax.loglog(times, run['relaxation.x'][1:], 'o', markersize=2, color=np.array(color) / 255, label=f'order {order}')
    ax.loglog(times, exact, '-', linewidth=0.8, color='black')
    if order < 1:
        ax.loglog(times[50:], times[50:] ** -order / math.gamma(1 - order), ':', color=np.array(color) / 255)
ax.set_ylim(1e-4, 1.5)
ax.set_xlabel('time t')
ax.set_ylabel('x(t)')
ax.set_title('Simulated relaxation on E_alpha(-t^alpha) (black)')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
