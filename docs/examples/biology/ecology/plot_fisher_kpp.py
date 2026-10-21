"""
# Fisher-KPP

A population that grows and diffuses invades the empty part of a line as a travelling front: behind it the density
is at capacity, ahead of it the line is empty, and the front keeps its shape while it moves at the constant speed
`2 sqrt(r D)`.
"""

# %%
# The model is created by name from the catalog with the left 5% of a line of 200 units occupied. The blueprint shows
# the space-time history of the density, where the front is a straight diagonal edge.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'fisher_kpp'

system = create(MODEL, length=800, D=1.0, r=1.0, spacing=0.25, front=0.05)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=default_blueprint(system), interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=80)

# %%
# Speed of the front
# ------------------
# Tracking the position where the density crosses one half, the front should move along a straight line whose slope
# approaches `2 sqrt(r D) = 2` once its shape has settled.

trajectory = simulation.trajectory_
density = trajectory['line.u']
positions = np.arange(density.shape[1]) * 0.25
front = np.array([positions[np.argmax(row < 0.5)] for row in density])
moving = (trajectory.times > 20) & (front < positions[-1] - 10)
speed = np.polyfit(trajectory.times[moving], front[moving], 1)[0]
fig, (profiles, position) = plt.subplots(1, 2, figsize=(11, 4))
for frame in range(0, len(trajectory.times), 10):
    profiles.plot(positions, density[frame], color=plt.cm.viridis(frame / len(trajectory.times)))
profiles.set_xlabel('position')
profiles.set_ylabel('density u')
profiles.set_title('The front keeps its shape as it moves')
position.plot(trajectory.times, front, label='simulated front')
position.plot(
    trajectory.times,
    front[moving][0] + 2 * (trajectory.times - trajectory.times[moving][0]),
    '--',
    color='black',
    label='speed 2 sqrt(r D)',
)
position.set_xlabel('time')
position.set_ylabel('position of u = 1/2')
position.set_title(f'Fitted speed {speed:.2f}, predicted 2')
position.legend()
fig.tight_layout()
plt.show()
