"""
# Rotating balls on rods

Two balls turning about a vertical axis spin up when they are pulled closer to it, like a skater drawing in the arms:
their angular momentum is conserved, so halving the distance quadruples the angular velocity.
"""

# %%
# The balls turn at 1 radian per unit time at distance 1 from the axis, and a schedule moves them to distance 0.5 at
# `t = 5`. The default blueprint shows the angles, which lie in `[0, 2π)`: the teeth of the sawtooth become four times
# narrower when the balls are pulled in.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rotating_balls_on_rods'


def radius_at(time):
    """Return the distance 1 until `t = 5` and 0.5 afterwards."""
    return 1.0 if time < 5.0 else 0.5


system = create(MODEL, omega=1.0, r=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
schedule = {'right.r': radius_at, 'left.r': radius_at}
simulation = Simulation(system, dt=0.04, observers=[observer], schedule=schedule).run(until=10)

# %%
# Angular velocity against the radius
# -----------------------------------
# With the angular momentum `H = m ω r^2` fixed, the angular velocity is `ω0 r0^2 / r^2`. Runs that pull the balls in
# to different distances at `t = 5` should turn at 1 before and at `1 / r^2`, dashed, after.

fig, ax = plt.subplots(figsize=(6, 4))
for radius in [0.9, 0.7, 0.5]:
    pulled = {'right.r': lambda time, value=radius: 1.0 if time < 5.0 else value}
    run = Simulation(create(MODEL, omega=1.0, r=1.0), dt=0.02, schedule=pulled).run(until=10)
    times = np.asarray(run.trajectory_.times)
    angle = np.unwrap(np.ravel(run.trajectory_['right.theta']))
    line = ax.plot(times[1:], np.diff(angle) / np.diff(times), label=f'r = {radius} after t = 5')[0]
    ax.hlines(1 / radius**2, 5, 10, linestyles='--', colors=line.get_color())
ax.set_xlabel('time')
ax.set_ylabel('angular velocity')
ax.set_title('Pulling the balls in spins them up as 1 / r²')
ax.legend()
plt.show()
