"""
# Logistic map

As the growth rate `r` rises from 2.8 to 4, the long-run value of `x` splits into 2, 4, 8, ... branches at ever closer
values of `r`, and past `r ≈ 3.57` it fills whole bands: the period-doubling route to chaos, with windows of order
such as the period-3 window near `r = 3.83`.
"""

# %%
# The map runs at the chaotic rate `r = 3.7`, and the model logs `x` as a time series. The orbit diagram, the values
# that `x` visits after a transient of 500 steps at 400 values of `r`, is logged once as points, and the blueprint puts
# it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov_spectrum, orbit_diagram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'logistic_map'

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/orbit_diagram',
            name='orbit diagram',
            visual_bounds=rrb.VisualBounds2D(x_range=[275, 405], y_range=[-102, 2]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x at r = 3.7'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, r=3.7), dt=1, observers=[observer]).run(until=200)
rates = np.linspace(2.8, 4.0, 400)
diagram = orbit_diagram(MODEL, 'map.r', rates, 'map.x', transient=500, keep=100, dt=1)
points = np.array(
    [(100 * r, -100 * x) for r, values in zip(diagram.values, diagram.points, strict=True) for x in values],
)
observer.recording_.log('orbit_diagram', rr.Points2D(points, radii=0.12, colors=[90, 170, 255]), static=True)
observer.recording_.flush()

# %%
# Lyapunov exponent along the cascade
# -----------------------------------
# The largest Lyapunov exponent should touch 0 at every period doubling, stay negative on the periodic branches and in
# the periodic windows, and turn positive where the orbit diagram fills bands, reaching `ln 2` at `r = 4`.

sample = np.linspace(2.8, 4.0, 41)
exponents = [lyapunov_spectrum(simulation.system, until=1000, parameters={'map.r': r})[0] for r in sample]
fig, (orbits, lyapunov) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
orbits.plot(points[:, 0] / 100, -points[:, 1] / 100, ',', color='C0')
orbits.set_ylabel('long-run x')
orbits.set_title('Orbit diagram and Lyapunov exponent of the logistic map')
lyapunov.plot(sample, exponents, '.-', color='C3')
lyapunov.axhline(0, color='black', linewidth=0.8)
lyapunov.axhline(np.log(2), color='black', linestyle=':', label='ln 2')
lyapunov.set_ylim(-1.5, 1)
lyapunov.set_xlabel('growth rate r')
lyapunov.set_ylabel('Lyapunov exponent')
lyapunov.legend(loc='lower left')
fig.tight_layout()
plt.show()
