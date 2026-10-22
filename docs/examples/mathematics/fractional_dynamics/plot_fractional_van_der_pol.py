"""
# Fractional Van der Pol

The fractional Van der Pol oscillator of order 0.9 still settles on a self-sustained oscillation: orbits from inside
and from outside wind onto one closed cycle in the phase plane. The memory of the fractional derivatives shrinks the
cycle as the order falls, from the amplitude 2 of the ordinary oscillator.
"""

# %%
# The model logs `x` and `y` as time series. After the run, its orbit from outside the cycle and an orbit from near the
# origin are logged once as lines in the phase plane, and the blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fractional_van_der_pol'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane of x and y',
            visual_bounds=rrb.VisualBounds2D(x_range=[-90, 90], y_range=[-90, 90]),
        ),
        rrb.TimeSeriesView(origin='/oscillator', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(create(MODEL, x=3.0), dt=0.1, observers=[observer]).run(until=40)
inside = Simulation(create(MODEL, x=0.1), dt=0.1).run(until=40).trajectory_
strips = [
    np.column_stack([25 * run['oscillator.x'], -25 * run['oscillator.y']]) for run in (simulation.trajectory_, inside)
]
observer.recording_.log(
    'phase_plane',
    rr.LineStrips2D(strips, radii=0.6, colors=[[255, 170, 40], [90, 170, 255]]),
    static=True,
)
observer.recording_.flush()

# %%
# Amplitude of the cycle
# ----------------------
# The amplitude of `x` on the cycle should be 2 for order 1, the ordinary Van der Pol oscillator with `mu = 1`, and
# shrink steadily as the order falls, while orbits from inside and outside reach the same amplitude.


def cycle_amplitude(start, order):
    """Return the largest displacement over the last 10 time units of a run of 40."""
    run = Simulation(create(MODEL, x=start, order=order), dt=0.1).run(until=40).trajectory_
    return np.max(np.abs(run['oscillator.x'][-100:]))


orders = np.linspace(0.7, 1.0, 7)
amplitudes = {start: [cycle_amplitude(start, order) for order in orders] for start in (0.1, 3.0)}
fig, (plane, amplitude) = plt.subplots(1, 2, figsize=(10, 4.5))
for strip, label in zip(strips, ('from outside', 'from inside'), strict=True):
    plane.plot(strip[:, 0] / 25, -strip[:, 1] / 25, label=label, linewidth=0.8)
plane.set_xlabel('x')
plane.set_ylabel('y')
plane.set_title('Both orbits wind onto one cycle, order 0.9')
plane.legend(fontsize=8)
for start, marker in ((0.1, 'o'), (3.0, 'x')):
    amplitude.plot(orders, amplitudes[start], marker, label=f'from x = {start}')
amplitude.axhline(2.0, color='black', linewidth=0.8, label='ordinary oscillator')
amplitude.set_xlabel('order alpha')
amplitude.set_ylabel('amplitude of x on the cycle')
amplitude.set_title('The memory shrinks the cycle')
amplitude.legend(fontsize=8)
fig.tight_layout()
plt.show()
