"""
# Optimal velocity system

Twenty cars start in uniform flow on a ring road, with one car slightly displaced. The uniform flow is unstable: the
small disturbance grows into a stop-and-go wave, where cars alternately crawl through a jam and race to the next.
"""

# %%
# The system is built from its class with 20 cars on a ring of length 40, the sensitivity `a = 1`, and the
# safety distance `hc = 2`, so that the uniform headway 2 is unstable, and runs for 200 time units. The blueprint shows
# the speeds of all cars in one time series view, without a legend.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.traffic import OptimalVelocitySystem

SYSTEM = 'optimal_velocity'

speeds = rrb.TimeSeriesView(
    origin='/cars/v',
    name='speeds',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
    plot_legend=rrb.PlotLegend(visible=False),
)
blueprint = rrb.Blueprint(
    speeds,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = OptimalVelocitySystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, observers=[observer])

# %%
# Instability of uniform flow
# ---------------------------
# With `V(h) = V0 (tanh(h - hc) + tanh(hc))`, uniform flow at the headway `b` is unstable when `V'(b) > a / 2`. At
# `b = hc = 2`, `V'(b) = V0 = 1 > 0.5`, so the spread of speeds should grow from the initial disturbance and saturate
# in a stop-and-go wave. A ring with a larger headway, where `V'(b) < a / 2`, should instead return to uniform flow.

stable = simulate(OptimalVelocitySystem(length=80.0))
fig, (spread, cycle) = plt.subplots(1, 2, figsize=(10, 4))
spread.semilogy(trajectory.times, trajectory.values('cars.v').std(axis=1) + 1e-12, label='headway 2, unstable')
spread.semilogy(stable.times, stable.values('cars.v').std(axis=1) + 1e-12, label='headway 4, stable')
spread.set_xlabel('time')
spread.set_ylabel('spread of speeds')
spread.set_title('The disturbance grows or decays')
spread.legend()
headways = np.mod(np.roll(trajectory.values('cars.x'), -1, axis=1) - trajectory.values('cars.x'), 40.0)
late = trajectory.times > 150
cycle.plot(
    headways[late].ravel(),
    trajectory.values('cars.v')[late].ravel(),
    '.',
    markersize=2,
    label='all cars, t > 150',
)
grid = np.linspace(0.0, 5.0, 200)
cycle.plot(grid, np.tanh(grid - 2.0) + np.tanh(2.0), color='black', label='V(h)')
cycle.set_xlabel('headway h')
cycle.set_ylabel('speed v')
cycle.set_title('Stop-and-go loop around V(h)')
cycle.legend()
fig.tight_layout()
plt.show()
