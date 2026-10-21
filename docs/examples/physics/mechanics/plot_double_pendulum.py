"""
# Double pendulum

A pendulum hanging from the bob of another, released from high up, tumbles chaotically: the lower bob traces a tangled,
never repeating path, and two releases that differ by a hair soon move in completely different ways.
"""

# %%
# The rods start high, at 2 and 2.5 radians from the downward vertical, and at rest. The default blueprint draws
# the rods, the bobs, and the fading trace of the lower bob.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'double_pendulum'


def bobs(theta1, theta2):
    """Return the positions of the two bobs of unit rods."""
    first = np.array([np.sin(theta1), np.cos(theta1)])
    return first, first + np.array([np.sin(theta2), np.cos(theta2)])


system = create(MODEL, theta1=2.0, theta2=2.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=10)
simulation = Simulation(system, dt=0.005, observers=[observer]).run(until=12)

# %%
# Sensitive dependence on the initial angle
# -----------------------------------------
# A second double pendulum starts with its upper angle larger by `10^-8` radians. The distance between the lower bobs
# of the two pendulums should grow exponentially, a straight line on the logarithmic axis, until it is as large as the
# pendulum itself.

runs = [
    Simulation(create(MODEL, theta1=np.pi / 2 + shift, theta2=np.pi / 2), dt=0.005).run(until=20)
    for shift in [0.0, 1e-8]
]
lower = [
    bobs(np.ravel(run.trajectory_['pendulum.theta1']), np.ravel(run.trajectory_['pendulum.theta2']))[1] for run in runs
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(runs[0].trajectory_.times, np.linalg.norm(lower[0] - lower[1], axis=0))
ax.set_xlabel('time')
ax.set_ylabel('distance between the lower bobs')
ax.set_title('Two releases 1e-8 radians apart')
plt.show()
