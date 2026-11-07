"""
# Lozi map

The Lozi map replaces the parabola of the Hénon map with a corner, `x' = 1 + y - a |x|`, so its strange attractor is
built of straight segments folded at sharp angles instead of curved bands.
"""

# %%
# The map iterates 3000 steps from the origin with Lozi's parameters `a = 1.7` and `b = 0.5`, and the system logs `x`
# and `y` as time series. After the run, the orbit is logged once more as points in the plane, which the blueprint
# shows first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'lozi_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/attractor',
            name='attractor',
            visual_bounds=rrb.VisualBounds2D(x_range=[-140, 140], y_range=[-70, 70]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=12)
simulation = Simulation(create_system(SYSTEM), dt=1, observers=[observer]).prepare().run(until=3000)
points = np.column_stack([simulation.trajectory_.values('map.x'), -simulation.trajectory_.values('map.y')])[10:]
observer.recording_.log('attractor', rr.Points2D(100 * points, radii=0.5, colors=[90, 170, 255]), static=True)
observer.recording_.flush()

# %%
# Lyapunov exponents along the slope
# ----------------------------------
# The map is piecewise linear, so its Jacobian has determinant `-b` everywhere and the two exponents should always sum
# to `ln b`. The largest exponent should be positive over a wide band of slopes `a`, near 0.47 at `a = 1.7`.

slopes = np.linspace(1.2, 1.8, 13)
spectra = np.array(
    [lyapunov(simulation.system, until=5000, parameters={'map.a': a}).exponents for a in slopes],
)
fig, (attractor, exponent_axes) = plt.subplots(1, 2, figsize=(10, 4))
attractor.plot(points[:, 0], -points[:, 1], ',', color='C0')
attractor.set_xlabel('x')
attractor.set_ylabel('y')
attractor.set_title('Attractor at a = 1.7, b = 0.5')
exponent_axes.plot(slopes, spectra[:, 0], 'o-', label='largest')
exponent_axes.plot(slopes, spectra.sum(axis=1), 's-', label='sum')
exponent_axes.axhline(np.log(0.5), color='black', linestyle=':', label='ln b')
exponent_axes.set_xlabel('slope a')
exponent_axes.set_ylabel('Lyapunov exponent')
exponent_axes.legend()
fig.tight_layout()
plt.show()
