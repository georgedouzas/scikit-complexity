"""
# Delayed car following

The lead car speeds up and slows down gently, and every follower reacts to the car ahead one second late. The delay
amplifies the oscillation from car to car, so the last cars of the chain swing their speeds far more than the leader.
"""

# %%
# The model is created by name from the catalog with eight cars, a lead car oscillating by 10% around speed 1 at
# `omega = 0.5`, and the reaction time `T = 1`, and runs for 60 time units. The blueprint shows the speeds of all cars
# in one time series view.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_car_following'

speeds = rrb.TimeSeriesView(
    origin='/',
    contents=[f'/car_{car}/v' for car in range(8)],
    name='speeds',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    speeds,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=60)

# %%
# Growth along the chain
# ----------------------
# For the linear model the oscillation grows from car to car when `omega < 2 lambda sin(omega T)`. With `lambda = 1`,
# `T = 1`, and `omega = 0.5`, `2 sin(0.5) = 0.96 > 0.5`, so the amplitude should grow along the chain. A shorter
# reaction time `T = 0.2` gives `0.40 < 0.5`, so the amplitude should decay instead.

fig, ax = plt.subplots(figsize=(6, 4.5))
for delay, label in [(1.0, 'T = 1, unstable chain'), (0.2, 'T = 0.2, stable chain')]:
    run = Simulation(create(MODEL, T=delay), dt=0.5).run(until=120).trajectory_
    late = run.times > 60
    amplitudes = [np.ptp(run[f'car_{car}.v'][late]) / 2 for car in range(8)]
    ax.semilogy(range(8), amplitudes, 'o-', label=label)
ax.set_xlabel('car in the chain (0 is the leader)')
ax.set_ylabel('amplitude of the speed oscillation')
ax.set_title('The delay amplifies the oscillation along the chain')
ax.legend()
plt.show()
