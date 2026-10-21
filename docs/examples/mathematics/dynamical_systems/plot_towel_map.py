"""
# Rössler's towel map

The orbit of this three-dimensional map fills a crumpled sheet, a towel, that is stretched in two directions and
squeezed in the third at every step: hyperchaos, with two positive Lyapunov exponents instead of one.
"""

# %%
# The map runs for 6000 steps from its default start, and the model logs `x`, `y`, and `z` as time series. After the
# run, the orbit is logged once more as points in three dimensions, with `y` stretched four times to show the
# folds, and the blueprint puts this view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import kaplan_yorke_dimension, lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'towel_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
camera = rrb.EyeControls3D(position=(1.25, -0.95, 0.95), look_target=(0.55, 0.0, 0.55), eye_up=(0.0, 0.0, 1.0))
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', eye_controls=camera, line_grid=False),
        rrb.TimeSeriesView(origin='/map', name='x, y, z', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=24)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=6000)
trajectory = simulation.trajectory_
points = np.column_stack([trajectory[f'map.{name}'] for name in 'xyz'])[50:]
colors = plt.get_cmap('plasma')(points[:, 1] / np.ptp(points[:, 1]) - points[:, 1].min() / np.ptp(points[:, 1]))
observer.recording_.log('attractor', rr.Points3D(points * [1, 4, 1], radii=0.004, colors=colors[:, :3]), static=True)
observer.recording_.flush()

# %%
# Two positive exponents
# ----------------------
# The Lyapunov spectrum should have two positive exponents and one strongly negative one, about 0.4, 0.4, and
# -3.3, and the Kaplan-Yorke dimension they give should lie between 2 and 3, a sheet folded on itself.

exponents = lyapunov_spectrum(simulation.system, until=20000)
fig, (sheet, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
sheet.scatter(points[:, 0], points[:, 2], s=0.3, c=colors)
sheet.set_xlabel('x')
sheet.set_ylabel('z')
sheet.set_title('The towel seen along y')
spectrum.bar(range(1, 4), exponents, color=['C3', 'C3', 'C0'])
spectrum.axhline(0, color='black', linewidth=0.8)
spectrum.set_xticks(range(1, 4))
spectrum.set_xlabel('exponent index')
spectrum.set_ylabel('Lyapunov exponent per step')
spectrum.set_title(
    f'{exponents[0]:.2f}, {exponents[1]:.2f}, {exponents[2]:.2f}; dimension {kaplan_yorke_dimension(exponents):.2f}',
)
fig.tight_layout()
plt.show()
