"""
# Ikeda delay equation

With a delayed feedback `x' = a sin x(t - tau)`, the rest point at `π` holds while the product `a tau` stays below
`π / 2`, and beyond it the state overshoots and keeps oscillating, with swings that grow with the gain.
"""

# %%
# The equation runs from `x = 2` with `a = 1` and `tau = 2`, past the Hopf bifurcation, and the system logs `x` as a
# time series, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import IkedaSystem

SYSTEM = 'ikeda'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/ikeda', name='x', axis_x=whole_run),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=5)
trajectory = simulate(IkedaSystem(), observers=[observer])

# %%
# Bifurcation diagram
# -------------------
# At the delay `tau = 2`, the extremes of `x` over the last part of a long run should coincide at `π` up to the gain
# `a = π / 4`, and split into the two ends of an oscillation that widens as the gain grows beyond it.

gains = np.linspace(0.2, 2.0, 37)
low, high = [], []
for a in gains:
    x = simulate(IkedaSystem(a=a), until=300).get_values('ikeda.x')[-500:]
    low.append(x.min())
    high.append(x.max())
fig, (series, diagram) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, trajectory.get_values('ikeda.x'))
series.axhline(np.pi, color='black', linestyle=':')
series.set_xlabel('time')
series.set_ylabel('x')
series.set_title('a = 1, tau = 2')
diagram.plot(gains, low, '.', color='C0')
diagram.plot(gains, high, '.', color='C0')
diagram.axvline(np.pi / 4, color='C3', linestyle=':', label='a tau = π / 2')
diagram.set_xlabel('gain a')
diagram.set_ylabel('extremes of x')
diagram.set_title('Hopf bifurcation at tau = 2')
diagram.legend()
fig.tight_layout()
plt.show()
