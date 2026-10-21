"""
# Bird swarm

Thirty self-propelled birds, each driven to the same speed and held together by a potential that attracts far and
repels near, form a cohesive swarm that organizes itself into a rotating mill or a flock flying in one direction.
"""

# %%
# The model is created by name from the catalog with thirty birds started at random. The observer saves every frame
# with the default blueprint, whose primary view shows the birds in the plane.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import milling_order, polar_order
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bird_swarm'

system = create(MODEL, n_birds=30, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Mill or flock
# -------------
# The polar order, the length of the mean heading, is one for a flock, and the milling order, the mean normalized
# angular momentum about the centre, is one for a mill. As the swarm organizes, one of the two should approach one,
# and every bird should approach the preferred speed `sqrt(alpha / beta) = sqrt(2)`.

trajectory = simulation.trajectory_
positions, velocities = trajectory['birds.x'], trajectory['birds.v']
fig, (order, path) = plt.subplots(1, 2, figsize=(11, 4))
order.plot(trajectory.times, polar_order(velocities, vectors=True), label='polar order')
order.plot(trajectory.times, np.abs(milling_order(positions, velocities)), label='|milling order|')
order.plot(trajectory.times, np.linalg.norm(velocities, axis=2).mean(axis=1) / np.sqrt(2), label='speed / sqrt(2)')
order.set_ylim(0, 1.1)
order.set_xlabel('time')
order.set_ylabel('order')
order.set_title('The swarm organizes')
order.legend()
path.plot(positions[-40:, :, 0], positions[-40:, :, 1], linewidth=0.7)
path.plot(positions[-1, :, 0], positions[-1, :, 1], 'o', color='black', markersize=3)
path.set_aspect('equal')
path.set_xlabel('x')
path.set_ylabel('y')
path.set_title('Tracks over the last 20 time units')
fig.tight_layout()
plt.show()
