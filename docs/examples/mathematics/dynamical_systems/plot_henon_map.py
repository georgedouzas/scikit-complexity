"""
# Hénon map

The orbit jumps around the plane without repeating and, after a few steps, every point lands on a thin curved band
folded on itself: the Hénon strange attractor, whose bands split into finer bands at every magnification.
"""

# %%
# The map iterates 3000 steps from the origin with the classic parameters `a = 1.4` and `b = 0.3`. The model logs `x`
# and `y` as time series. After the run, the whole orbit is logged once more as points in the plane, and the blueprint
# puts that view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import correlation_dimension
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'henon_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/attractor',
            name='attractor',
            visual_bounds=rrb.VisualBounds2D(x_range=[-140, 140], y_range=[-60, 60]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=12)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=3000)
points = np.column_stack([simulation.trajectory_['map.x'], -simulation.trajectory_['map.y']])[10:]
observer.recording_.log('attractor', rr.Points2D(100 * points, radii=0.5, colors=[90, 170, 255]), static=True)
observer.recording_.flush()

# %%
# Self-similar bands and fractal dimension
# ----------------------------------------
# Zooming into a small box of the attractor should show the same pattern of three bands, one of them double, as the
# whole picture. The correlation sum of the orbit should grow as a power of the radius with the correlation dimension
# close to the 1.21 that Grassberger and Procaccia measured.

orbit = np.column_stack([simulation.trajectory_['map.x'], simulation.trajectory_['map.y']])[10:]
long_orbit = Simulation(create(MODEL), dt=1).run(until=100_000).trajectory_
long_orbit = np.column_stack([long_orbit['map.x'], long_orbit['map.y']])[10:]
dimension = correlation_dimension(long_orbit[:20000])
fig, (whole, zoom) = plt.subplots(1, 2, figsize=(10, 4))
whole.plot(orbit[:, 0], orbit[:, 1], ',', color='C0')
whole.add_patch(plt.Rectangle((0.55, 0.15), 0.15, 0.05, fill=False, color='C3'))
whole.set_xlabel('x')
whole.set_ylabel('y')
whole.set_title(f'Attractor, correlation dimension {dimension.dimension:.2f}')
box = (long_orbit[:, 0] > 0.55) & (long_orbit[:, 0] < 0.7) & (long_orbit[:, 1] > 0.15) & (long_orbit[:, 1] < 0.2)
zoom.plot(long_orbit[box, 0], long_orbit[box, 1], '.', markersize=1, color='C3')
zoom.set_xlabel('x')
zoom.set_ylabel('y')
zoom.set_title('The red box of 100000 steps: bands within bands')
fig.tight_layout()
plt.show()
