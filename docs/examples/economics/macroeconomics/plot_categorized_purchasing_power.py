"""
# Categorized purchasing power

Forrest splits the demand and the supply of money of a national economy into the sectors of basic, desired, and
luxurious living conditions. A policy aimed at one sector moves all of them, but a feedback component of market
reactions can decouple the sectors, so each index answers only to its own policy.
"""

# %%
# A unit step of policy hits the luxury sector at time 1, with the illustrative matrices of the system, for 15 time
# units. The default blueprint shows the gaps between demand and supply of the three sectors under feedback.

import matplotlib.pyplot as plt

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'categorized_purchasing_power'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
decoupled = create_simulation(SYSTEM).set_params(dt=0.05, observers=[observer]).prepare().run(until=15)
coupled = create_simulation(SYSTEM, feedback=False).set_params(dt=0.05).prepare().run(until=15)

# %%
# Spillovers with and without feedback
# ------------------------------------
# Without feedback the policy of the luxury sector should spill over into the basic and the desired indices. With the
# feedback component of Eq. 22.12, the basic and desired indices should stay at zero while the luxury index rises to
# the steady state of its own decoupled system.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axis, simulation, title in ((axes[0], coupled, 'Without feedback'), (axes[1], decoupled, 'With feedback')):
    trajectory = simulation.trajectory_
    for index, name in enumerate(('basic', 'desired', 'luxury'), start=1):
        axis.plot(trajectory.times, trajectory.values(f'economy.y{index}'), label=name)
    axis.set_xlabel('time')
    axis.set_title(title)
    axis.legend()
axes[0].set_ylabel('index y of the sector')
fig.tight_layout()
plt.show()
