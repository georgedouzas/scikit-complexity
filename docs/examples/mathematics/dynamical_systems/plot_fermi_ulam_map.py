"""
# Fermi-Ulam map

A ball bounces between a fixed wall and a wall that oscillates in time. Each bounce changes the speed by the speed of
the moving wall, and the time to the next bounce depends on the new speed: at low speeds the motion is chaotic, and at
high speeds invariant curves bound it, so the ball is never accelerated without limit.
"""

# %%
# One orbit iterates 1000 bounces from `u = 0.5`, and the system logs `u` and `θ` as time series. Orbits from a grid of
# starting speeds and phases are logged once as points in the plane of phase and speed, which the blueprint shows
# first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import FermiUlamMapSystem

SYSTEM = 'fermi_ulam_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_space',
            name='phase space',
            visual_bounds=rrb.VisualBounds2D(x_range=[-5, 640], y_range=[-160, 5]),
        ),
        rrb.TimeSeriesView(origin='/map', name='u, θ', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=5)
trajectory = simulate(FermiUlamMapSystem(), observers=[observer])
orbits = []
for u in np.linspace(0.1, 1.5, 12):
    for theta in np.linspace(0.0, 2 * np.pi, 4, endpoint=False):
        trajectory = simulate(FermiUlamMapSystem(u=u, theta=theta), until=400)
        orbits.append(np.column_stack([trajectory.get_values('map.theta'), trajectory.get_values('map.u')]))
points = np.concatenate(orbits)
points = points[points[:, 1] < 1.6]
observer.recording_.log(
    'phase_space',
    rr.Points2D(100 * points * [1, -1], radii=0.4, colors=[90, 170, 255]),
    static=True,
)
observer.recording_.flush()

# %%
# The chaotic sea grows with the amplitude
# ----------------------------------------
# The chaotic orbit from the slowest start should reach higher speeds when the wall moves more, up to the first
# invariant curve that spans all phases, and the fixed points at `u = 1 / m` should sit inside islands.

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for axis, epsilon in zip(axes, (0.05, 0.1, 0.2), strict=True):
    for u in np.linspace(0.05, 1.5, 15):
        trajectory = simulate(FermiUlamMapSystem(u=u, theta=1.0, epsilon=epsilon), until=500)
        axis.plot(trajectory.get_values('map.theta'), trajectory.get_values('map.u'), ',')
    axis.set_ylim(0, 1.6)
    axis.set_xlabel('phase θ')
    axis.set_title(f'ε = {epsilon}')
axes[0].set_ylabel('speed u')
fig.tight_layout()
plt.show()
