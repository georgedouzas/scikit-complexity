"""
# Social force

Helbing and Molnar's social force model of pedestrians: every walker accelerates towards its desired velocity and is
pushed away from the walkers around it. Two crowds walking in opposite directions along a corridor start mixed and
bump into each other, but they soon organise themselves into lanes of walkers moving the same way, which lets both
crowds walk at almost their desired speed.
"""

# %%
# Forty walkers head right and forty head left at the desired speed of 1.34 m/s along a 10 m corridor that is
# periodic along its length, starting at random positions. The default blueprint shows the walkers after a minute,
# sorted into lanes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'social_force'
LENGTH, SPEED, UNTIL = 10.0, 1.34, 60.0

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
system = create(MODEL, n_walkers=40, length=LENGTH, speed=SPEED, random_state=0)
simulation = Simulation(system, dt=0.2, solver=RK4(dt=0.05), observers=[observer], random_state=0).run(until=UNTIL)

# %%
# Lane order and walking speed
# ----------------------------
# Cutting the corridor into ten bands across its width, the lane order `sum_b n_b ((r_b - l_b) / n_b)^2 / N`, with
# `r_b` and `l_b` the walkers of band `b` moving right and left, is about `1 / n_b`, small, for mixed crowds and 1
# when every band holds a single direction. It should rise within tens of seconds as lanes form, and the mean
# velocity along the walking direction should approach the desired speed as the collisions stop. In the densest
# corridor the lanes should keep breaking up and the crowds should stay well below their desired speed.


def lane_order(rightward, leftward, bands=10):
    """Return the lane order of the walkers from their lateral positions."""
    edges = np.linspace(0, LENGTH, bands + 1)
    right = np.histogram(rightward[:, 1], edges)[0]
    left = np.histogram(leftward[:, 1], edges)[0]
    total = right + left
    occupied = total > 0
    return float(np.sum((right - left)[occupied] ** 2 / total[occupied]) / total.sum())


fig, (order, speed) = plt.subplots(1, 2, figsize=(10, 4))
for index, walkers in enumerate((20, 40, 60)):
    system = create(MODEL, n_walkers=walkers, length=LENGTH, speed=SPEED, random_state=0)
    trajectory = Simulation(system, dt=0.2, solver=RK4(dt=0.05), random_state=0).run(until=UNTIL).trajectory_
    rightward, leftward = trajectory['rightward.x'], trajectory['leftward.x']
    order.plot(
        trajectory.times,
        [lane_order(right, left) for right, left in zip(rightward, leftward, strict=True)],
        color=f'C{index}',
        label=f'{walkers} each way',
    )
    velocity = (trajectory['rightward.v'][:, :, 0].mean(axis=1) - trajectory['leftward.v'][:, :, 0].mean(axis=1)) / 2
    speed.plot(trajectory.times, velocity, color=f'C{index}', label=f'{walkers} each way')
order.set_xlabel('time (s)')
order.set_ylabel('lane order')
order.set_title('Lanes form')
order.legend()
speed.axhline(SPEED, color='grey', linestyle=':', label='desired speed')
speed.set_xlabel('time (s)')
speed.set_ylabel('mean velocity along the walking direction (m/s)')
speed.set_title('Lanes let the crowds walk freely')
speed.legend()
fig.tight_layout()
plt.show()
