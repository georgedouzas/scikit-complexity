"""
# Koopman slow manifold

A quadratic flow with a fast and a slow direction: every trajectory first drops quickly onto a parabola, the slow
manifold, and then slides along it to the origin. The phase plane shows trajectories from points above and below the
parabola that bend sharply onto it and run together toward the origin.
"""

# %%
# The model is created by name from the catalog with `mu = -0.05` and `lam = -1`, starting from `(1, -1)`. Six more
# runs from other starting points share the phase plane. After the runs, every orbit is logged once as a line in the
# phase plane together with the parabola `x2 = b x1^2`, `b = lam / (lam - 2 mu)`, in dark blue, and the blueprint puts
# this view first, as the primary view. The vertical axis of the plane points up.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'koopman_slow_manifold'
MU, LAM = -0.05, -1.0
SCALE = 100.0

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane',
            visual_bounds=rrb.VisualBounds2D(x_range=[-170, 170], y_range=[-170, 120]),
        ),
        rrb.TimeSeriesView(origin='/flow', name='x1, x2', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(create(MODEL), dt=0.1, observers=[observer]).run(until=40)
starts = [(-1.0, -1.0), (1.5, 0.0), (-1.5, 0.0), (0.5, 1.5), (-0.5, 1.5), (1.2, 1.5)]
orbits = [simulation.trajectory_] + [
    Simulation(create(MODEL, x1=x1, x2=x2), dt=0.1).run(until=40).trajectory_ for x1, x2 in starts
]
b = LAM / (LAM - 2 * MU)
parabola = np.linspace(-1.6, 1.6, 101)
strips = [np.column_stack([SCALE * orbit['flow.x1'], -SCALE * orbit['flow.x2']]) for orbit in orbits]
observer.recording_.log('phase_plane/orbits', rr.LineStrips2D(strips, radii=1.0, colors=[255, 170, 40]), static=True)
observer.recording_.log(
    'phase_plane/manifold',
    rr.LineStrips2D([np.column_stack([SCALE * parabola, -SCALE * b * parabola**2])], radii=1.0, colors=[60, 90, 200]),
    static=True,
)
observer.recording_.flush()

# %%
# Linear dynamics in Koopman coordinates
# --------------------------------------
# With the observable `x1^2`, the coordinates `(x1, x2, x1^2)` evolve linearly with the eigenvalues `mu`, `lam`, and
# `2 mu`. The Koopman eigenfunctions `x1` and `x2 - b x1^2` should therefore decay exactly as `exp(mu t)` and
# `exp(lam t)` along every orbit, straight lines on a logarithmic scale.

times = simulation.trajectory_.times
fig, (plane, eigen) = plt.subplots(1, 2, figsize=(10, 4))
for orbit in orbits:
    plane.plot(orbit['flow.x1'], orbit['flow.x2'], color='tab:orange', linewidth=1)
plane.plot(parabola, b * parabola**2, color='tab:blue', label=f'slow manifold x2 = {b:.3f} x1^2')
plane.set_xlabel('x1')
plane.set_ylabel('x2')
plane.set_title('Fast collapse onto the slow manifold')
plane.legend()
for orbit in orbits[:3]:
    eigen.semilogy(times, np.abs(orbit['flow.x1']), color='tab:orange', linewidth=1)
    eigen.semilogy(times, np.abs(orbit['flow.x2'] - b * orbit['flow.x1'] ** 2), color='tab:green', linewidth=1)
eigen.semilogy(times, np.exp(MU * times), 'k--', label='exp(mu t)')
eigen.semilogy(times, np.exp(LAM * times), 'k:', label='exp(lam t)')
eigen.set_ylim(1e-12, 3)
eigen.set_xlabel('time')
eigen.set_ylabel('|eigenfunction|')
eigen.set_title('Koopman eigenfunctions decay exponentially')
eigen.legend()
fig.tight_layout()
plt.show()
