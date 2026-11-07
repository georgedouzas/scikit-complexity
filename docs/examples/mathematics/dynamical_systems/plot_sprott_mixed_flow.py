"""
# Sprott's mixed flow

The same three equations hold a conservative torus and a dissipative strange attractor side by side: from `(1, 0, 0)`
the orbit winds on a torus that keeps every volume, and from `(2, 0, 0)` it falls onto a chaotic attractor that
shrinks volumes.
"""

# %%
# The orbit from `(2, 0, 0)` runs for 200 time units, and the system logs its three states as time series. Both orbits
# are logged once as lines in three dimensions, and the blueprint puts that view first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'sprott_mixed_flow'
STARTS = {'torus': (1.0, [80, 160, 240]), 'attractor': (2.0, [230, 80, 80])}

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/orbits', name='torus and attractor', line_grid=False),
        rrb.TimeSeriesView(origin='/flow', name='x, y, z', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=20)
simulation = Simulation(create_system(SYSTEM), dt=0.05, observers=[observer]).prepare().run(until=200)
runs = {}
for name, (x, color) in STARTS.items():
    runs[name] = Simulation(create_system(SYSTEM, x=x), dt=0.05).prepare().run(until=1000).trajectory_
    points = np.column_stack([runs[name].values(f'flow.{state}') for state in 'xyz'])[:4000]
    observer.recording_.log(f'orbits/{name}', rr.LineStrips3D([points], radii=0.005, colors=color), static=True)
observer.recording_.flush()

# %%
# Volume rate
# -----------
# Volumes change at the rate `2(y + z)`, the trace of the Jacobian. Its running mean along each orbit should tend to 0
# on the torus and to a negative value on the attractor (Datseris and Parlitz, Exercise 8.16).

fig, axis = plt.subplots(figsize=(7, 4))
for name, trajectory in runs.items():
    rate = 2 * (trajectory.values('flow.y') + trajectory.values('flow.z'))
    running = np.cumsum(rate) / np.arange(1, rate.size + 1)
    axis.plot(trajectory.times, running, label=f'{name}, mean {rate.mean():.3f}')
axis.axhline(0, color='black', linestyle=':')
axis.set_xlabel('time')
axis.set_ylabel('running mean of 2(y + z)')
axis.set_title('Conservative torus, dissipative attractor')
axis.legend()
fig.tight_layout()
plt.show()
