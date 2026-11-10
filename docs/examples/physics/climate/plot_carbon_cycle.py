"""
# Excitable carbon cycle

Pushed far enough below its steady state, the dissolved carbon of the ocean does not recover at once: the carbonate ion
concentration first collapses, then overshoots to more than twice its steady value in a large excursion before the
ocean returns to rest, like the carbon cycle disruptions recorded in the geological past.
"""

# %%
# The example builds `CarbonCycleSystem` starting at the steady carbonate ion concentration `c = 83.58` with the
# dissolved inorganic carbon lowered to `w = 2050`, and runs for 20 time units. The blueprint shows only `c`, since `w`
# is about 25 times larger and would flatten the excursion.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import CarbonCycleSystem

SYSTEM = 'carbon_cycle'

carbonate = rrb.TimeSeriesView(
    origin='/ocean/c',
    name='carbonate ion',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    carbonate,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = CarbonCycleSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
trajectory = simulate(system, observers=[observer])

# %%
# A threshold for the excursion
# -----------------------------
# Excitability means that the response is not proportional to the perturbation. Lowering the initial dissolved carbon
# by increasing amounts, the peak of the carbonate ion concentration should stay near its steady value for small
# perturbations and jump to a large excursion once the perturbation crosses a threshold.

starts = np.linspace(2050.0, 2190.0, 29)
peaks = []
fig, (runs, response) = plt.subplots(1, 2, figsize=(10, 4))
for w0 in starts:
    run = simulate(CarbonCycleSystem(w=w0))
    peaks.append(run.get_values('ocean.c').max())
    if w0 in starts[::7]:
        runs.plot(run.times, run.get_values('ocean.c'), linewidth=1, label=f'w0 = {w0:.0f}')
runs.set_xlabel('time')
runs.set_ylabel('carbonate ion c (μmol/kg)')
runs.set_title('Small and large perturbations')
runs.legend()
response.plot(2197 - starts, peaks, 'o-', markersize=3)
response.set_xlabel('initial deficit of dissolved carbon 2197 - w0')
response.set_ylabel('peak carbonate ion c')
response.set_title('The excursion switches on at a threshold')
fig.tight_layout()
plt.show()
