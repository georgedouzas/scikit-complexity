"""
# Tent map

Above slope 1 the tent map is chaotic at once, with no period-doubling cascade: the long-run values fill bands that
split in two at ever closer slopes as `mu` falls towards 1, and merge into a single band over
`[mu (1 - mu / 2), mu / 2]` above `mu = √2`.
"""

# %%
# The map runs at the slope `mu = 1.99`, and the system logs `x` as a time series. The orbit diagram, the values that
# `x` visits after a transient of 500 steps at 400 slopes, is logged once as points, and the blueprint puts it first, as
# the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import OrbitDiagramExperiment
from skcomplex.systems import create_system, describe

SYSTEM = 'tent_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/orbit_diagram',
            name='orbit diagram',
            visual_bounds=rrb.VisualBounds2D(x_range=[95, 205], y_range=[-102, 2]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x at mu = 1.99', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create_system(SYSTEM), dt=1, observers=[observer]).prepare().run(until=200)
slopes = np.linspace(1.0, 2.0, 400)
diagram = (
    OrbitDiagramExperiment(
        describe(SYSTEM).factory,
        'map.mu',
        slopes,
        'map.x',
        transient=500,
        keep=100,
        dt=1,
    )
    .prepare()
    .run()
)
points = np.array(
    [(100 * mu, -100 * x) for mu, values in zip(diagram.values, diagram.points_, strict=True) for x in values],
)
observer.recording_.log('orbit_diagram', rr.Points2D(points, radii=0.12, colors=[90, 170, 255]), static=True)
observer.recording_.flush()

# %%
# Lyapunov exponent
# -----------------
# The map stretches every interval by `mu` at each step, so its Lyapunov exponent should equal `ln mu` at every slope,
# positive for all `mu > 1`, unlike the logistic map, whose exponent dips below 0 in periodic windows.

sample = np.linspace(1.05, 1.99, 20)
exponents = [lyapunov(simulation.system, until=2000, parameters={'map.mu': mu}).exponents[0] for mu in sample]
fig, (orbits, exponent_axes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
orbits.plot(points[:, 0] / 100, -points[:, 1] / 100, ',', color='C0')
orbits.set_ylabel('long-run x')
orbits.set_title('Orbit diagram and Lyapunov exponent of the tent map')
exponent_axes.plot(slopes, np.log(slopes), color='black', label='ln mu')
exponent_axes.plot(sample, exponents, 'o', color='C3', label='Benettin estimate')
exponent_axes.set_xlabel('slope mu')
exponent_axes.set_ylabel('Lyapunov exponent')
exponent_axes.legend()
fig.tight_layout()
plt.show()
