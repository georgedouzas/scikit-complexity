"""
# Pheromone foraging

Ants leave the nest at the left and search at random until they find the food at the right. Each ant that carries
food home lays pheromone on its way, other ants follow the scent, and a trail forms between nest and food: stigmergy,
coordination through marks left in the environment.
"""

# %%
# The model is created by name from the catalog with forty ants on a grid of 40 by 20 cells. The observer saves every
# third step with the default blueprint, whose primary view is the image of the pheromone.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pheromone_foraging'

system = create(MODEL, n_ants=40)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=600)

# %%
# The trail and the traffic
# -------------------------
# Once the trail forms, more ants should be carrying food at any time, and the pheromone should concentrate along the
# band between nest and food rather than spread over the grid.

trajectory = simulation.trajectory_
frames = range(len(trajectory.times))
carrying = [np.sum(trajectory.agents('colony/ants', frame).values['carrying']) for frame in frames]
pheromone = fields_of(trajectory)
fig, (image, traffic) = plt.subplots(1, 2, figsize=(11, 4))
image.imshow(pheromone[-1].T, cmap='magma', origin='lower', aspect='auto')
image.set_title('Pheromone at the end of the run')
image.set_xlabel('x (nest left, food right)')
image.set_ylabel('y')
traffic.plot(trajectory.times, carrying)
traffic.set_xlabel('step')
traffic.set_ylabel('ants carrying food')
traffic.set_title('Traffic rises as the trail forms')
fig.tight_layout()
plt.show()
