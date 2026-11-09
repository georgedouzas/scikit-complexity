"""
# Delayed car following

The lead car speeds up and slows down gently, and every follower reacts to the car ahead one second late. The delay
amplifies the oscillation from car to car, so the last cars of the chain swing their speeds far more than the leader.
"""

# %%
# The system is built from its class with eight cars, a lead car oscillating by 10% around speed 1 at
# `omega = 0.5`, and the reaction time `T = 1`, and runs for 60 time units. The blueprint shows the speeds of all cars
# in one time series view.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.traffic import DelayedCarFollowingSystem

SYSTEM = 'delayed_car_following'

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
system = DelayedCarFollowingSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, observers=[observer])

# %%
# Growth along the chain
# ----------------------
# For the linear system the oscillation grows from car to car when `omega < 2 lambda sin(omega T)`. With `lambda = 1`,
# `T = 1`, and `omega = 0.5`, `2 sin(0.5) = 0.96 > 0.5`, so the amplitude should grow along the chain. A shorter
# reaction time `T = 0.2` gives `0.40 < 0.5`, so the amplitude should decay instead.

fig, ax = plt.subplots(figsize=(6, 4.5))
for delay, label in [(1.0, 'T = 1, unstable chain'), (0.2, 'T = 0.2, stable chain')]:
    run = simulate(DelayedCarFollowingSystem(T=delay), until=120)
    late = run.times > 60
    amplitudes = [np.ptp(run.get_values(f'car_{car}.v')[late]) / 2 for car in range(8)]
    ax.semilogy(range(8), amplitudes, 'o-', label=label)
ax.set_xlabel('car in the chain (0 is the leader)')
ax.set_ylabel('amplitude of the speed oscillation')
ax.set_title('The delay amplifies the oscillation along the chain')
ax.legend()
plt.show()
