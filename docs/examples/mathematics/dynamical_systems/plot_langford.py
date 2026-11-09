"""
# Langford attractor

The orbit circles the `z` axis on a torus whose surface wrinkles and tears, so the motion is no longer quasiperiodic:
Langford's flow at the reference parameters of Dilão is chaotic on a doughnut-shaped strange attractor.
"""

# %%
# The flow runs for 200 time units, and the system logs its three states as time series. After the run, the orbit is
# logged once more as a line in three dimensions, and the blueprint puts that view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import LangfordSystem

SYSTEM = 'langford'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', line_grid=False),
        rrb.TimeSeriesView(origin='/langford', name='x, y, z', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=20)
trajectory = simulate(LangfordSystem(), dt=0.01, observers=[observer])
points = np.column_stack([trajectory.get_values(f'langford.{state}') for state in 'xyz'])[2000:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.004, colors=[230, 120, 60]), static=True)
observer.recording_.flush()

# %%
# Torus with and without its symmetry-breaking term
# -------------------------------------------------
# Without the term `f z x³` the flow is symmetric under rotations about the `z` axis, so the radius and the height obey
# a planar flow that settles on a cycle, and the successive maxima of `z` settle on one value. With `f = 0.1` the
# maxima scatter along a folded curve: the torus has broken.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for axis, f in zip(axes, (0.0, 0.1), strict=True):
    run = simulate(LangfordSystem(f=f), until=600, dt=0.01)
    z = run.get_values('langford.z')[5000:]
    maxima = z[find_peaks(z)[0]]
    axis.plot(maxima[:-1], maxima[1:], '.', markersize=3)
    axis.set_xlabel('maximum $z_n$')
    axis.set_ylabel('next maximum $z_{n+1}$')
    axis.set_title(f'Return map of the maxima of z, f = {f}')
fig.tight_layout()
plt.show()
