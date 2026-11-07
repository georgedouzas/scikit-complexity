"""
# Hyperchaotic Rössler flow

Rössler's four-dimensional flow stretches its attractor in two directions at once: two Lyapunov exponents are positive,
so nearby orbits separate along a surface rather than a line, the first example of hyperchaos.
"""

# %%
# The flow runs for 300 time units, and the system logs its four states as time series. After the run, the orbit is
# logged once more as a line in the space of `x`, `y`, and `w`, and the blueprint puts that view first.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'hyperchaotic_rossler'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='x, y, w', line_grid=False),
        rrb.TimeSeriesView(origin='/rossler', name='x, y, z, w', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=20)
simulation = Simulation(create_system(SYSTEM), dt=0.05, observers=[observer]).prepare().run(until=300)
points = np.column_stack([simulation.trajectory_.values(f'rossler.{state}') for state in 'xyw'])[400:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.05, colors=[200, 90, 200]), static=True)
observer.recording_.flush()

# %%
# Lyapunov spectrum
# -----------------
# Averaged over longer and longer times, the four exponents should settle near 0.11 and 0.02, two positive values, then
# a zero exponent along the flow and a strongly negative one.

times = np.array([250, 500, 1000, 2000, 4000])
spectra = np.array(
    [lyapunov(simulation.system, until=until).exponents[:3] for until in times],
)
fig, (orbit, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(points[:, 0], points[:, 2], linewidth=0.3)
orbit.set_xlabel('x')
orbit.set_ylabel('w')
orbit.set_title('Projection of the attractor')
for index in range(3):
    spectrum.semilogx(times, spectra[:, index], 'o-', label=f'exponent {index + 1}')
spectrum.axhline(0, color='black', linestyle=':')
spectrum.set_xlabel('averaging time')
spectrum.set_ylabel('Lyapunov exponent')
spectrum.set_title('Two positive exponents')
spectrum.legend()
fig.tight_layout()
plt.show()
