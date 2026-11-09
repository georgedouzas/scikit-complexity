"""
# Zombie outbreak

A single zombie in a town of 500 humans bites its way through the population. Humans destroy zombies, but the dead
rise again, so whatever the humans do the outbreak ends in the doomsday state where no human is left.
"""

# %%
# The system is built from its class with the parameters of Munz et al. (2009), and the observer saves every other
# frame with the default blueprint, whose primary view is the time series of the three compartments.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import ZombieOutbreakSystem

SYSTEM = 'zombie_outbreak'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ZombieOutbreakSystem(), observers=[observer])

# %%
# Doomsday and the conserved total
# --------------------------------
# Without births every event moves an individual between compartments, so the total `S + Z + R` should stay at 501,
# while the humans fall to zero and the zombies take over.

times = trajectory.times
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, name in (('S.x', 'humans'), ('Z.x', 'zombies'), ('R.x', 'removed')):
    ax.plot(times, trajectory.get_values(label), label=name)
total = trajectory.get_values('S.x') + trajectory.get_values('Z.x') + trajectory.get_values('R.x')
ax.plot(times, total, '--', color='black', label='total')
ax.set_xlabel('time (days)')
ax.set_ylabel('individuals')
ax.set_title('Zombies overrun the humans, the total is conserved')
ax.legend()
plt.show()
