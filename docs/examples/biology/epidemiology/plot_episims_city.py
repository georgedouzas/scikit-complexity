"""
# EpiSims city

People of a synthetic city sleep in households and spend the day at workplaces, and infect the people they meet at
either. Households carry the disease within families and workplaces between them, so closing the workplaces confines
an outbreak to a few households.
"""

# %%
# The system is created by name from the catalog with a thousand people, and the observer saves every half day with the
# default blueprint, whose primary view is the network of locations with the people on it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'episims_city'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=200)

# %%
# Closing workplaces
# ------------------
# With the workplaces open the epidemic should reach most people, and with them closed it should stay in the households
# of the first cases.


def compartments(trajectory):
    health = np.array([trajectory.agents('people', frame).values['health'] for frame in range(201)])
    return [(health == state).mean(axis=1) for state in (0, 1, 2)]


closed = Simulation(create_system(SYSTEM, closed=True), random_state=0).prepare().run(until=200).trajectory_
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, trajectory, title in ((axes[0], simulation.trajectory_, 'workplaces open'), (axes[1], closed, 'closed')):
    for share, name in zip(compartments(trajectory), ('susceptible', 'infected', 'recovered'), strict=True):
        ax.plot(np.arange(201) / 2, share, label=name)
    ax.set_xlabel('day')
    ax.set_title(title)
axes[0].set_ylabel('share of people')
axes[0].legend()
fig.tight_layout()
plt.show()
