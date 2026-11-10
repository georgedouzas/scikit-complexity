"""
# Nosé-Hoover oscillator

A harmonic oscillator coupled to a thermostat conserves volume on average, and its state space is mixed: a periodic
orbit, tori that wrap around it, and a chaotic sea that fills the rest, as Datseris and Parlitz show with three
initial conditions.
"""

# %%
# The chaotic orbit from `(0, 0.1, 0)` runs for 200 time units, and the system logs its three states as time series.
# The periodic orbit from `(0, 1.5499, 0)` and a torus from `(0, 1, 0)` are logged once as lines in three dimensions,
# next to the chaotic one, and the blueprint puts that view first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import NoseHooverSystem

SYSTEM = 'nose_hoover'
STARTS = {'periodic': 1.549934227822898, 'torus': 1.0, 'chaotic': 0.1}
COLORS = {'periodic': [230, 80, 80], 'torus': [80, 160, 240], 'chaotic': [150, 150, 150]}

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/orbits', name='orbits', line_grid=False),
        rrb.TimeSeriesView(origin='/oscillator', name='chaotic x, y, z', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=20)
trajectory = simulate(NoseHooverSystem(), observers=[observer])
orbits = {}
for name, y in STARTS.items():
    trajectory = simulate(NoseHooverSystem(y=y))
    orbits[name] = np.column_stack([trajectory.get_values(f'oscillator.{state}') for state in 'xyz'])
    observer.recording_.log(
        f'orbits/{name}',
        rr.LineStrips3D([orbits[name]], radii=0.01, colors=COLORS[name]),
        static=True,
    )
observer.recording_.flush()

# %%
# A Poincaré section
# ------------------
# Every crossing of the plane `x = 0` upwards marks a point in `(y, z)`. The periodic orbit should leave one point, the
# torus a closed curve around it, and the chaotic orbit a scatter that avoids the curve. Since `z' = 1 - y²`, every
# bounded orbit has `<y²> = 1`.

fig, axis = plt.subplots(figsize=(6, 5))
for name, y in STARTS.items():
    trajectory = simulate(NoseHooverSystem(y=y), until=2000, dt=0.02)
    x, y_values, z = (trajectory.get_values(f'oscillator.{state}') for state in 'xyz')
    crossing = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    mean_square = np.mean(y_values**2)
    axis.plot(y_values[crossing], z[crossing], '.', markersize=2, label=f'{name}, <y²> = {mean_square:.3f}')
axis.set_xlabel('y')
axis.set_ylabel('z')
axis.set_title('Section x = 0 of three orbits')
axis.legend()
fig.tight_layout()
plt.show()
