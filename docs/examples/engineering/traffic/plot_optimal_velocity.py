"""
# Optimal velocity model

Twenty cars start in uniform flow on a ring road, with one car slightly displaced. The uniform flow is unstable: the
small disturbance grows into a stop-and-go wave, where cars alternately crawl through a jam and race to the next.
"""

# %%
# The model is created by name from the catalog with 20 cars on a ring of length 40, the sensitivity `a = 1`, and the
# safety distance `hc = 2`, so that the uniform headway 2 is unstable, and runs for 200 time units. The blueprint shows
# the speeds of all cars in one time series view, without a legend.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'optimal_velocity'

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
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=200)

# %%
# Instability of uniform flow
# ---------------------------
# With `V(h) = V0 (tanh(h - hc) + tanh(hc))`, uniform flow at the headway `b` is unstable when `V'(b) > a / 2`. At
# `b = hc = 2`, `V'(b) = V0 = 1 > 0.5`, so the spread of speeds should grow from the initial disturbance and saturate
# in a stop-and-go wave. A ring with a larger headway, where `V'(b) < a / 2`, should instead return to uniform flow.

trajectory = simulation.trajectory_
stable = Simulation(create(MODEL, length=80.0), dt=1.0).run(until=200).trajectory_
fig, (spread, cycle) = plt.subplots(1, 2, figsize=(10, 4))
spread.semilogy(trajectory.times, trajectory['cars.v'].std(axis=1) + 1e-12, label='headway 2, unstable')
spread.semilogy(stable.times, stable['cars.v'].std(axis=1) + 1e-12, label='headway 4, stable')
spread.set_xlabel('time')
spread.set_ylabel('spread of speeds')
spread.set_title('The disturbance grows or decays')
spread.legend()
headways = np.mod(np.roll(trajectory['cars.x'], -1, axis=1) - trajectory['cars.x'], 40.0)
late = trajectory.times > 150
cycle.plot(headways[late].ravel(), trajectory['cars.v'][late].ravel(), '.', markersize=2, label='all cars, t > 150')
grid = np.linspace(0.0, 5.0, 200)
cycle.plot(grid, np.tanh(grid - 2.0) + np.tanh(2.0), color='black', label='V(h)')
cycle.set_xlabel('headway h')
cycle.set_ylabel('speed v')
cycle.set_title('Stop-and-go loop around V(h)')
cycle.legend()
fig.tight_layout()
plt.show()
