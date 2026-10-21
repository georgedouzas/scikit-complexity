"""
# Newton-Raphson map

Newton's method for `z³ = 1` sends almost every starting point to one of the three cube roots of unity, but which root
it finds depends on the start in an intricate way: the three basins of attraction meet along a fractal boundary, where
every neighbourhood of a boundary point holds starts that go to all three roots.
"""

# %%
# Six runs of the model start near the basin boundaries and are iterated for 30 steps; the model logs their
# coordinates as time series. The basins over the square `[-1.5, 1.5]²` are computed with the same map in one NumPy
# pass over a 300 by 300 grid and logged once as an image, coloured by the root each start reaches, and the
# blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import box_counting_dimension
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'newton_raphson_map'
ROOTS = np.exp(2j * np.pi * np.arange(3) / 3)
SIZE = 300


def basins(size, steps=40):
    """Return the index of the root reached from every point of a grid, and the steps it took, by the same map."""
    axis = np.linspace(-1.5, 1.5, size)
    z = axis[None, :] + 1j * axis[::-1, None]
    z[z == 0] = 1e-9
    for _ in range(steps):
        z = 2 * z / 3 + 1 / (3 * z**2)
    return np.abs(z[..., None] - ROOTS).argmin(axis=-1)


# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/basins',
            name='basins of the three roots',
            visual_bounds=rrb.VisualBounds2D(x_range=[0, SIZE], y_range=[0, SIZE]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, x=-0.5, y=0.02), dt=1, observers=[observer]).run(until=30)
starts = [(-0.5, 0.02), (-0.5, -0.02), (0.3, 0.55), (0.3, 0.45), (0.3, -0.55), (0.3, -0.45)]
orbits = [Simulation(create(MODEL, x=x, y=y), dt=1).run(until=30).trajectory_ for x, y in starts]
labels = basins(SIZE)
palette = np.array([[230, 90, 60], [70, 160, 220], [250, 200, 60]], dtype=np.uint8)
observer.recording_.log('basins', rr.Image(palette[labels]), static=True)
observer.recording_.flush()

# %%
# Fractal basin boundary
# ----------------------
# Each orbit should end on the root of the basin it starts in, and two starts on either side of a boundary should go
# to different roots. The boundary of the basins should have a box-counting dimension strictly between 1 and 2, as a
# fractal curve, unlike the smooth boundary of a straight cut.

finals = [np.angle(run['map.x'][-1] + 1j * run['map.y'][-1], deg=True).round().astype(int) for run in orbits]
fine = basins(1024)
boundary = (np.diff(fine, axis=0, prepend=fine[:1]) != 0) | (np.diff(fine, axis=1, prepend=fine[:, :1]) != 0)
dimension = box_counting_dimension(boundary)
fig, (picture, edges) = plt.subplots(1, 2, figsize=(10, 4.5))
picture.imshow(palette[labels], extent=(-1.5, 1.5, -1.5, 1.5))
for run in orbits:
    picture.plot(run['map.x'], run['map.y'], 'k.-', markersize=3, linewidth=0.8)
picture.plot(ROOTS.real, ROOTS.imag, 'w*', markersize=10)
picture.set_xlabel('Re z')
picture.set_ylabel('Im z')
picture.set_xlim(-1.5, 1.5)
picture.set_ylim(-1.5, 1.5)
picture.set_title(f'Orbits end at angles {sorted(set(map(int, finals)))}°')
edges.imshow(boundary, cmap='gray_r', extent=(-1.5, 1.5, -1.5, 1.5))
edges.set_xlabel('Re z')
edges.set_ylabel('Im z')
edges.set_title(f'Basin boundary, box dimension {dimension:.2f}')
fig.tight_layout()
plt.show()
