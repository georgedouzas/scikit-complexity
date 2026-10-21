"""
# Mutualism

Two species that feed each other both end above the carrying capacity they would reach alone. The benefit saturates
with the handling time of the partner's resource, so the populations settle instead of growing without bound.
"""

# %%
# The model is created by name from the catalog with its default saturating mutualism and run from small
# populations. The observer saves every second frame with the default blueprint, whose primary view is the time series
# of both populations.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'mutualism'

system = create(MODEL, first=5.0, second=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=30)

# %%
# With and without the partner
# ----------------------------
# Running each species alone, with the other set to zero, gives logistic growth to its carrying capacity of 50. With
# the partner, both should settle clearly above that line.

alone = Simulation(create(MODEL, first=5.0, second=0.0), dt=0.1).run(until=30).trajectory_
trajectory = simulation.trajectory_
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, trajectory['first.n'], label='first, with partner')
ax.plot(trajectory.times, trajectory['second.n'], label='second, with partner')
ax.plot(alone.times, alone['first.n'], '--', color='black', label='first, alone')
ax.axhline(50.0, color='grey', linestyle=':', label='carrying capacity')
ax.set_xlabel('time')
ax.set_ylabel('population')
ax.set_title(f'Mutualists settle at {trajectory["first.n"][-1]:.0f} and {trajectory["second.n"][-1]:.0f}, above 50')
ax.legend()
plt.show()
