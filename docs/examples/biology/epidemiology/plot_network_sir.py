"""
# Network SIR

Travellers live in cities at the nodes of a network and move along its edges. One infected traveller starts an
epidemic that spreads within cities and is carried between them, so the infection reaches the connected hubs first
and the outbreak in the whole network rises, peaks, and burns out.
"""

# %%
# The model is created by name from the catalog on its default Barabási-Albert network of 20 cities with 20
# travellers each. The recorded run stops at day 25, near the peak of the outbreak, so that the default blueprint's view
# of the network shows the cities in the middle of the epidemic; the analysis runs the outbreak to its end.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'network_sir'

system = create(MODEL, residents=20, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=25)

# %%
# The outbreak in the network
# ---------------------------
# Counting the health states of the travellers, the infected should rise, peak, and decline while the recovered
# accumulate, as in a single well-mixed SIR epidemic but slowed by the travel between cities.

trajectory = Simulation(create(MODEL, residents=20, random_state=0), dt=1, random_state=0).run(until=150).trajectory_
health = np.array([trajectory.agents('travellers', frame).values['health'] for frame in range(len(trajectory.times))])
fig, ax = plt.subplots(figsize=(8, 4))
for value, label in [(0, 'susceptible'), (2, 'infected'), (3, 'recovered')]:
    ax.plot(trajectory.times, np.sum(health == value, axis=1), label=label)
ax.set_xlabel('step')
ax.set_ylabel('travellers')
ax.set_title(f'{np.sum(health[-1] == 0)} of {health.shape[1]} travellers never infected')
ax.legend()
plt.show()
