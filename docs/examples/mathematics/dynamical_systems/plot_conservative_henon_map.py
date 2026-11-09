"""
# Conservative Hénon map

Hénon's area-preserving map rotates the plane by `2π nu` and adds a quadratic kick. Around the centre at the origin
the orbits trace closed curves, island chains appear further out, and beyond the separatrix of the saddle the orbits
escape: the region of bounded orbits is the dynamic aperture of a particle accelerator.
"""

# %%
# One orbit iterates 1000 steps at Hénon's rotation `cos 2π nu = 0.24`, and the system logs `x` and `y` as time series.
# Orbits from 40 starting points along the `x` axis are logged once as points in the plane, which the blueprint shows
# first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.dynamical_systems import ConservativeHenonMapSystem

SYSTEM = 'conservative_henon_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_space',
            name='phase space',
            visual_bounds=rrb.VisualBounds2D(x_range=[-120, 120], y_range=[-120, 120]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=5)
trajectory = simulate(ConservativeHenonMapSystem(), observers=[observer])
orbits = []
for x in np.linspace(0.02, 0.8, 40):
    trajectory = simulate(ConservativeHenonMapSystem(x=x), until=500)
    orbit = np.column_stack([trajectory.get_values('map.x'), trajectory.get_values('map.y')])
    orbits.append(orbit[np.all(np.abs(orbit) < 1.2, axis=1)])
points = np.concatenate(orbits)
observer.recording_.log(
    'phase_space',
    rr.Points2D(100 * points * [1, -1], radii=0.3, colors=[90, 170, 255]),
    static=True,
)
observer.recording_.flush()

# %%
# Dynamic aperture
# ----------------
# For every rotation `nu`, the largest `x0` from which the orbit of `(x0, 0)` stays bounded for 1000 steps, found by
# bisection, is the dynamic aperture (Dilão, Exercise 3.13). It should shrink to 0 as `nu` approaches 0, where the map
# is globally unstable.


def bounded(x, nu):
    """Return whether the orbit of `(x, 0)` stays within 10 of the origin for 1000 steps."""
    trajectory = simulate(ConservativeHenonMapSystem(x=x, nu=nu))
    return bool(np.all(np.abs(trajectory.get_values('map.x')) < 10))


rotations = np.linspace(0.04, 0.46, 15)
apertures = []
for nu in rotations:
    low, high = 0.0, 2.0
    for _ in range(10):
        low, high = ((low + high) / 2, high) if bounded((low + high) / 2, nu) else (low, (low + high) / 2)
    apertures.append(low)
fig, (phase, aperture) = plt.subplots(1, 2, figsize=(10, 4.5))
for orbit in orbits:
    phase.plot(orbit[:, 0], orbit[:, 1], ',')
phase.set_xlabel('x')
phase.set_ylabel('y')
phase.set_title('Orbits at cos 2π nu = 0.24')
aperture.plot(rotations, apertures, 'o-')
aperture.set_xlabel('rotation nu')
aperture.set_ylabel('largest bounded x0')
aperture.set_title('Dynamic aperture')
fig.tight_layout()
plt.show()
