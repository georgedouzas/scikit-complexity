"""
# Thomas cyclically symmetric attractor

The orbit wanders through a chain of rounded cells that repeat under the cyclic swap of `x`, `y`, and `z`, the strange
attractor of Thomas's flow `x' = -a x + sin y` at the damping `a = 0.18`.
"""

# %%
# The flow runs for 500 time units, and the system logs its three states as time series. After the run, the whole
# orbit is logged once more as a line in three dimensions, and the blueprint puts that view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import ThomasSystem

SYSTEM = 'thomas'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', line_grid=False),
        rrb.TimeSeriesView(origin='/thomas', name='x, y, z', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=20)
trajectory = simulate(ThomasSystem(), observers=[observer])
points = np.column_stack([trajectory.get_values(f'thomas.{state}') for state in 'xyz'])[200:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.01, colors=[120, 200, 120]), static=True)
observer.recording_.flush()

# %%
# Damping and chaos
# -----------------
# As the damping `a` falls, the stable fixed point gives way to limit cycles and then to chaos, so the largest Lyapunov
# exponent should stay at or below 0 for strong damping and turn positive below about `a = 0.2`.

dampings = np.linspace(0.12, 0.32, 11)
largest = [lyapunov(ThomasSystem(), until=1000, parameters={'thomas.a': a}).exponents[0] for a in dampings]
fig, (orbit, exponent_axes) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(points[:, 0], points[:, 1], linewidth=0.3)
orbit.set_xlabel('x')
orbit.set_ylabel('y')
orbit.set_title('Projection of the attractor at a = 0.18')
exponent_axes.plot(dampings, largest, 'o-')
exponent_axes.axhline(0, color='black', linestyle=':')
exponent_axes.set_xlabel('damping a')
exponent_axes.set_ylabel('largest Lyapunov exponent')
exponent_axes.set_title('Chaos below a ≈ 0.2')
fig.tight_layout()
plt.show()
