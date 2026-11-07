"""
# Light-emitting robots with sensory delay

Each robot emits a Gaussian light, measures the total light of the others, and slows down in the light. The robots
react to the light with a delay. A robot that reacts late crosses dark regions fast and slows down only once it is in
the light of others, so the robots gather into clusters. A robot that predicts the light from its trend slows down
before it arrives and speeds up before it leaves, so the robots avoid each other and spread over the arena.
"""

# %%
# The system is created by name from the catalog with the values of Argun et al.: 50 robots in a periodic square of
# side 10, the intensity `I0 = 5`, the width `r0 = 1`, the speeds 1 in the dark and 0.1 in strong light, the time scale
# `tau = 2` of rotational diffusion, and a delay of 10 s. Every step lasts 0.5 s. The default blueprint shows the
# robots in the arena, colored by the light they measure.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'light_emitting_robots'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# Delay against prediction
# ------------------------
# With 50 robots the arena is so bright that every robot moves near its slowest speed, so the comparison uses 20
# robots. After 300 s, the robots should sit in clusters with a delay of 10 s, loosely with no delay, and spread out
# with a prediction 10 s ahead, as in figure 10.6 of Argun et al.

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, delay in zip(axes, (10.0, 0.0, -10.0), strict=True):
    trajectory = create_simulation(SYSTEM, n_robots=20, delay=delay).prepare().run(until=600).trajectory_
    snapshot = trajectory.agents('robots', 600)
    positions = np.asarray(snapshot.locations)
    ax.scatter(positions[:, 0], positions[:, 1], c=snapshot.values['light'], vmin=0, vmax=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(f'delay {delay:+.0f} s')
plt.show()
