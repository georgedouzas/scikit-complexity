"""
# Fractional damped oscillator

With no friction at all, `gamma = 0`, the oscillator of order 0.8 still loses its amplitude: the memory of the
fractional derivatives damps it, and the orbit spirals into the origin of the phase plane. The displacement follows
the Mittag-Leffler function `E_(2 alpha)(-omega^2 t^(2 alpha))`, the solution of the fractional oscillation equation.
"""

# %%
# The model logs `x` and `v` as time series. After the run, its orbit in the phase plane of `x` and `v` is logged once
# as a line, next to the closed orbit of order 1, and the blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import mittag_leffler
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fractional_oscillator'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane of x and v',
            visual_bounds=rrb.VisualBounds2D(x_range=[-110, 110], y_range=[-110, 110]),
        ),
        rrb.TimeSeriesView(origin='/oscillator', name='x, v', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(create(MODEL, order=0.8), dt=0.1, observers=[observer]).run(until=30)
harmonic = Simulation(create(MODEL, order=1.0), dt=0.1).run(until=2 * np.pi).trajectory_
strips = [
    np.column_stack([100 * run['oscillator.x'], -100 * run['oscillator.v']])
    for run in (harmonic, simulation.trajectory_)
]
observer.recording_.log(
    'phase_plane',
    rr.LineStrips2D(strips, radii=0.8, colors=[[90, 170, 255], [255, 170, 40]]),
    static=True,
)
observer.recording_.flush()

# %%
# Mittag-Leffler oscillation
# --------------------------
# The Laplace transform of the two equations with `v(0) = 0` gives `x(t) = x(0) E_(2 alpha)(-omega^2 t^(2 alpha))`.
# The simulated displacement of every order should lie on it, oscillating with an amplitude that decays as a power
# law for `alpha < 1` and keeping its amplitude for order 1.

fig, ax = plt.subplots(figsize=(7, 4.5))
for order, color in ((1.0, 'tab:blue'), (0.9, 'tab:green'), (0.8, 'tab:orange'), (0.6, 'tab:red')):
    run = Simulation(create(MODEL, order=order), dt=0.1).run(until=30).trajectory_
    exact = mittag_leffler(2 * order, 1.0, -(run.times ** (2 * order)))
    ax.plot(run.times, run['oscillator.x'], 'o', markersize=1.5, color=color, label=f'order {order}')
    ax.plot(run.times, exact, '-', linewidth=0.8, color='black')
ax.set_xlabel('time t')
ax.set_ylabel('x(t)')
ax.set_title('Simulated oscillation on E_(2 alpha)(-t^(2 alpha)) (black)')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
