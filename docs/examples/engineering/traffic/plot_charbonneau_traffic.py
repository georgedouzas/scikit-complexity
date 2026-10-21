"""
# Charbonneau traffic

Cars on a one-way road speed up to the limit while the road ahead is clear and brake hard when they come too close.
Now and then one driver brakes for no reason, and the cars behind brake in turn: a jam forms and travels backwards
along the road while the cars themselves keep moving forwards.
"""

# %%
# One hundred cars start at rest with random spacings. The default blueprint plots the position of every car against
# time, a space-time diagram in which a car in a jam runs flat for a while.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'charbonneau_traffic'

system = create(MODEL, probability=0.1, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=200)

# %%
# Jams travel backwards
# ---------------------
# In the space-time diagram every car is a line rising to the right, and slow cars are dark. Jams should show as dark
# bands that drift to the left, against the traffic, as cars leave a jam at its front and join it at its back.

trajectory = simulation.trajectory_
frames = range(len(trajectory))
positions = np.array([trajectory.agents('cars', frame).values['x'] for frame in frames])
speeds = np.array([trajectory.agents('cars', frame).values['v'] for frame in frames])
times = np.repeat(np.asarray(trajectory.times)[:, None], positions.shape[1], axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
points = ax.scatter(positions, times, c=speeds, s=0.5, cmap='viridis')
ax.set_xlabel('position along the road')
ax.set_ylabel('step')
ax.set_title('Jams drift backwards while the cars drive on')
fig.colorbar(points, label='speed')
plt.show()
