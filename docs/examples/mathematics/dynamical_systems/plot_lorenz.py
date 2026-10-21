"""
# Lorenz

The trajectory never settles and never repeats: it winds around two unstable fixed points and switches between the two
wings of the butterfly-shaped Lorenz attractor at irregular times.
"""

# %%
# The model logs its three states as time series. After the run, the whole trajectory is logged once more as a line in
# three dimensions, and the blueprint puts that view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lorenz'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
camera = rrb.EyeControls3D(position=(0.0, -62.0, 26.0), look_target=(0.0, 0.0, 24.0), eye_up=(0.0, 0.0, 1.0))
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', eye_controls=camera, line_grid=False),
        rrb.TimeSeriesView(
            origin='/lorenz',
            name='x, y, z',
            axis_x=whole_run,
            defaults=[rr.SeriesLines.from_fields(widths=1.5)],
        ),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=20)
simulation = Simulation(create(MODEL), dt=0.01, observers=[observer]).run(until=50)
points = np.column_stack([simulation.trajectory_[f'lorenz.{state}'] for state in 'xyz'])
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.15, colors=[255, 170, 40]), static=True)
observer.recording_.flush()

# %%
# The Lorenz map
# --------------
# Lorenz showed that each maximum of `z` is almost a function of the previous one. Plotting successive maxima of the
# simulated `z` against each other should give a thin tent-shaped curve, the signature of chaos on the attractor.

z = simulation.trajectory_['lorenz.z']
maxima = z[find_peaks(z)[0]]
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(maxima[:-1], maxima[1:], '.', markersize=4)
ax.set_xlabel('maximum $z_n$')
ax.set_ylabel('next maximum $z_{n+1}$')
ax.set_title(f'Lorenz map of {len(maxima)} maxima of z')
plt.show()
