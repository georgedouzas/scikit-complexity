"""
# Circle map

The average rotation per step, the winding number, locks onto rational values `p / q` over whole intervals of the bare
rotation `Omega`, the Arnold tongues. At the critical nonlinearity `K = 1` the plateaus fill the interval up to a set of
measure zero and the winding number climbs a devil's staircase.
"""

# %%
# The map runs at `Omega = 0.25`, `K = 0.9`, inside the tongue of winding number 1 / 4, and the model logs the angle as
# a time series. The winding number at `K = 1` over 201 values of `Omega`, averaged over 500 steps after a transient of
# 100, is logged once as points, and the blueprint puts this staircase first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import orbit_diagram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'circle_map'


def winding_numbers(rotations, K):
    """Return the mean rotation per step of the lifted map after a transient, for every bare rotation."""
    diagram = orbit_diagram(create(MODEL, K=K), 'map.Omega', rotations, 'map.theta', transient=100, keep=500, dt=1)
    return np.array(
        [
            np.mean(Omega - K / (2 * np.pi) * np.sin(2 * np.pi * theta))
            for Omega, theta in zip(rotations, diagram.points, strict=True)
        ],
    )


blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/staircase',
            name='winding number at K = 1',
            visual_bounds=rrb.VisualBounds2D(x_range=[-3, 103], y_range=[-103, 3]),
        ),
        rrb.TimeSeriesView(origin='/map', name='angle'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=100)
rotations = np.linspace(0.0, 1.0, 201)
critical = winding_numbers(rotations, K=1.0)
points = np.column_stack([100 * rotations, -100 * critical])
observer.recording_.log('staircase', rr.Points2D(points, radii=0.6, colors=[255, 170, 40]), static=True)
observer.recording_.flush()

# %%
# Mode locking grows with the nonlinearity
# ----------------------------------------
# Without nonlinearity the winding number should equal `Omega`. As `K` grows, plateaus at `0`, `1 / 2`, `1 / 3`,
# `1 / 4` and other rationals should widen, and the share of `Omega` on a plateau should
# grow steadily with `K`. It tends to 1 at `K = 1` only in the limit of a fine grid and long averages, since most of the
# locked measure there sits in very narrow tongues; this grid of 201 values catches about half of it.

strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
windings = {K: critical if K == 1.0 else winding_numbers(rotations, K=K) for K in strengths}
fig, (staircases, locked) = plt.subplots(1, 2, figsize=(11, 4))
for K in (0.0, 0.5, 1.0):
    staircases.plot(rotations, windings[K], label=f"K = {K}")
for value in (1 / 4, 1 / 3, 1 / 2):
    staircases.axhline(value, color='gray', linewidth=0.5)
staircases.set_xlabel(r'bare rotation $\Omega$')
staircases.set_ylabel('winding number')
staircases.set_title("Devil's staircase")
staircases.legend()
shares = [np.mean(np.abs(np.diff(windings[K])) < 1e-3) for K in strengths]
locked.plot(strengths, shares, 'o-')
locked.set_xlabel('nonlinearity K')
locked.set_ylabel(r'share of $\Omega$ on a plateau')
locked.set_title('Arnold tongues widen with K')
fig.tight_layout()
plt.show()
