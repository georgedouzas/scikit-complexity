"""
# Doubling map

One hundred points start in an interval of width 0.01 and the doubling map spreads them over the unit interval within
about eight steps. They keep covering it evenly for the whole run, long after floating-point doubling would have sent
every point to 0.
"""

# %%
# The ensemble iterates the map for 100 steps on the fractions `k / q` with the default prime denominator `q`, where the
# model applies `k' = 2k (mod q)` exactly. After the run, the cells of a grid of 25
# bins of `x` (rows, 0 at the bottom) that hold a point at each of the first 30 steps (columns) are logged once as an
# image, a thin line that fans out to fill the column, and the blueprint puts it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'doubling_map'

system = create(MODEL, n_points=100, random_state=0)
# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/spreading',
            name='where the points are, step by step',
            visual_bounds=rrb.VisualBounds2D(x_range=[0, 360], y_range=[0, 300]),
        ),
        rrb.TimeSeriesView(origin='/points', name='x', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=100)
occupied = [np.histogram(row, bins=25, range=(0, 1))[0] > 0 for row in simulation.trajectory_['points.x'][:30]]
image = np.where(np.array(occupied).T[::-1, :, None], [255, 170, 40], [20, 20, 30]).astype(np.uint8)
observer.recording_.log('spreading', rr.Image(image.repeat(12, axis=0).repeat(12, axis=1)), static=True)
observer.recording_.flush()

# %%
# Exact orbits against floating-point doubling
# --------------------------------------------
# Doubling a float and dropping its integer part shifts out one of its 53 binary digits, so the same starting points
# iterated with `2x mod 1` in floating point should all reach exactly 0 after about 55 steps. The model's points should
# stay spread, with a flat histogram at the uniform invariant density `p(x) = 1` of Bertin's Exercise 7.2.

values = simulation.trajectory_['points.x']
naive = [values[0]]
for _ in range(len(values) - 1):
    naive.append(np.mod(2 * naive[-1], 1.0))
fig, (collapse, density) = plt.subplots(1, 2, figsize=(10, 4))
collapse.plot(simulation.trajectory_.times, np.mean(np.array(naive) == 0, axis=1), label='floating-point $2x$ mod 1')
collapse.plot(simulation.trajectory_.times, np.mean(values == 0, axis=1), label='model, exact on $k / q$')
collapse.set_xlabel('step')
collapse.set_ylabel('fraction of points at 0')
collapse.set_title('Floating-point doubling collapses to 0')
collapse.legend()
density.hist(values[20:].ravel(), bins=20, range=(0, 1), density=True, label='model, last 80 steps')
density.axhline(1.0, color='black', label='invariant density $p(x) = 1$')
density.set_ylim(0, 1.5)
density.set_xlabel('x')
density.set_ylabel('density')
density.set_title('The exact orbits stay uniform')
density.legend(loc='lower center')
fig.tight_layout()
plt.show()
