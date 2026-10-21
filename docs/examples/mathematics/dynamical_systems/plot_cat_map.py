"""
# Arnold's cat map

A small square of starting points is stretched along one direction and squeezed along the other at every step, and
wrapped around the torus, so that within a few steps it is drawn into thin parallel stripes that cover the whole unit
square: a mixing map that keeps area.
"""

# %%
# The model runs from one point for 200 steps and logs its coordinates as time series. A grid of 1600 starting points
# in the square `[0.4, 0.6]²`, iterated for 3 steps, is logged once as points coloured by their starting `x`, and the
# blueprint puts this picture of the mixing first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cat_map'
STEPS = 3

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/mixing',
            name='square after 3 steps',
            visual_bounds=rrb.VisualBounds2D(x_range=[-2, 102], y_range=[-102, 2]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=200)
grid = np.linspace(0.4, 0.6, 40)
starts = np.array([(x, y) for x in grid for y in grid])
runs = [Simulation(create(MODEL, x=x, y=y), dt=1).run(until=6).trajectory_ for x, y in starts]
x = np.array([run['map.x'] for run in runs])
y = np.array([run['map.y'] for run in runs])
colors = plt.get_cmap('plasma')((starts[:, 0] - 0.4) / 0.2)[:, :3]
observer.recording_.log(
    'mixing',
    rr.Points2D(np.column_stack([100 * x[:, STEPS], -100 * y[:, STEPS]]), radii=0.5, colors=colors),
    static=True,
)
observer.recording_.flush()

# %%
# Stretching and mixing
# ---------------------
# The map is linear with the eigenvalues `φ²` and `1 / φ²`, `φ` the golden mean, so its Lyapunov exponents should be
# `± 2 ln φ ≈ ± 0.962` and the square should be stretched about 2.6 times per step. Once the stripes are thinner than
# a cell, the share of the 100 cells of a 10 by 10 grid that the points occupy should approach 1.

exponents = lyapunov_spectrum(simulation.system, until=2000)
cells = [len(set(zip((10 * x[:, step]).astype(int), (10 * y[:, step]).astype(int), strict=True))) for step in range(7)]
fig, (snapshots, occupied) = plt.subplots(1, 2, figsize=(10, 4.5))
for step in (0, 1, STEPS):
    snapshots.scatter(x[:, step], y[:, step], s=1, c=colors)
snapshots.set_xlim(0, 1)
snapshots.set_ylim(0, 1)
snapshots.set_xlabel('x')
snapshots.set_ylabel('y')
snapshots.set_title(f'Steps 0, 1, 3; exponents {exponents[0]:.3f}, {exponents[1]:.3f}')
occupied.plot(range(7), np.array(cells) / 100, 'o-')
occupied.set_xlabel('step')
occupied.set_ylabel('share of cells occupied')
occupied.set_title('The square spreads over the torus')
fig.tight_layout()
plt.show()
