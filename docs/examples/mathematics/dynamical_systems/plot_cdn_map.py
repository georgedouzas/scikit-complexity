"""
# Cascais-Dilão-Noronha map

The Gaussian map `x' = exp(-alpha² x²) + β` of a nonlinear circuit grows more complex as the shift `β` falls, and then
simpler again: period doubling into chaos is followed by reverse period doubling back to a fixed point.
"""

# %%
# The map iterates 200 steps at `alpha = 2.8` and `β = -0.5`, inside the chaotic window, and the system logs `x` as a
# time series. The orbit diagram over `β` is logged once as points, and the blueprint puts it first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import OrbitDiagramExperiment
from skcomplex.systems.mathematics.dynamical_systems import CDNMapSystem

SYSTEM = 'cdn_map'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/orbit_diagram',
            name='orbit diagram',
            visual_bounds=rrb.VisualBounds2D(x_range=[-102, 2], y_range=[-102, 52]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x at β = -0.5', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(CDNMapSystem(), observers=[observer])
shifts = np.linspace(-1.0, 0.0, 400)
diagram = (
    OrbitDiagramExperiment(CDNMapSystem, 'map.beta', shifts, 'map.x', transient=500, keep=100, dt=1).prepare().run()
)
points = np.array(
    [(100 * beta, -100 * x) for beta, values in zip(diagram.values, diagram.points_, strict=True) for x in values],
)
observer.recording_.log('orbit_diagram', rr.Points2D(points, radii=0.12, colors=[90, 170, 255]), static=True)
observer.recording_.flush()

# %%
# Lyapunov exponent
# -----------------
# The Lyapunov exponent should be positive over most of `β` between about -0.8 and -0.25, the chaotic band of Dilão's
# Fig. 3.17, with dips below 0 in the periodic windows, and negative outside it.

sample = np.linspace(-1.0, 0.0, 101)
exponents = [lyapunov(CDNMapSystem(), until=3000, parameters={'map.beta': beta}).exponents[0] for beta in sample]
fig, (orbits, exponent_axes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
orbits.plot(points[:, 0] / 100, -points[:, 1] / 100, ',', color='C0')
orbits.set_ylabel('long-run x')
orbits.set_title('Orbit diagram and Lyapunov exponent at alpha = 2.8')
exponent_axes.plot(sample, exponents, color='C3')
exponent_axes.axhline(0, color='black', linestyle=':')
exponent_axes.set_xlabel('shift β')
exponent_axes.set_ylabel('Lyapunov exponent')
fig.tight_layout()
plt.show()
