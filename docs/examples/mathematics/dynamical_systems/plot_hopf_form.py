"""
# Hopf normal form

Past the supercritical Hopf bifurcation, `mu > 0`, the focus at the origin repels and every orbit spirals onto a
limit cycle of radius `√(-mu / a)`, from inside and from outside alike: a stable oscillation born with zero amplitude
that grows as the square root of the distance to the bifurcation.
"""

# %%
# The run starts near the origin at `x = 0.1` with `mu = 0.25` and `a = -1`, and spirals out onto the circle of radius
# 0.5. The model logs `x` and `y` as time series. After the run, its orbit and an orbit started outside the cycle are
# logged once as lines in the phase plane, and the blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hopf_form'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane',
            visual_bounds=rrb.VisualBounds2D(x_range=[-110, 110], y_range=[-110, 110]),
        ),
        rrb.TimeSeriesView(origin='/form', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(create(MODEL, x=0.1), dt=0.1, observers=[observer]).run(until=50)
outside = Simulation(create(MODEL, x=1.0), dt=0.1).run(until=50).trajectory_
strips = [np.column_stack([100 * run['form.x'], -100 * run['form.y']]) for run in (simulation.trajectory_, outside)]
observer.recording_.log(
    'phase_plane',
    rr.LineStrips2D(strips, radii=0.8, colors=[[255, 170, 40], [90, 170, 255]]),
    static=True,
)
observer.recording_.flush()

# %%
# Amplitude of the cycle
# ----------------------
# The amplitude reached after a long run should be 0 for `mu < 0`, where the focus is stable, and `√mu` for `mu > 0`
# with `a = -1`.

rates = np.linspace(-0.2, 0.5, 15)
amplitudes = [np.hypot(*Simulation(create(MODEL, x=0.3, mu=mu), dt=0.1).run(until=300).state_) for mu in rates]
fig, (plane, amplitude) = plt.subplots(1, 2, figsize=(10, 4.5))
for strip, label in zip(strips, ('from inside', 'from outside'), strict=True):
    plane.plot(strip[:, 0] / 100, -strip[:, 1] / 100, label=label)
plane.set_aspect('equal')
plane.set_xlabel('x')
plane.set_ylabel('y')
plane.set_title('Both orbits wind onto the cycle')
plane.legend(fontsize=8)
grid = np.linspace(0, 0.5, 100)
amplitude.plot(grid, np.sqrt(grid), color='black', label=r'$\sqrt{\mu}$')
amplitude.plot([-0.2, 0], [0, 0], color='black')
amplitude.plot(rates, amplitudes, 'o', label='simulated')
amplitude.set_xlabel(r'$\mu$')
amplitude.set_ylabel('final amplitude')
amplitude.set_title('Amplitude grows as the square root of mu')
amplitude.legend()
fig.tight_layout()
plt.show()
