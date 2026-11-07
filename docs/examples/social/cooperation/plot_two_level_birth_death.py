"""
# Two-level birth-death process

The finite process behind the two-level replicator equation: groups of twenty defectors and altruistic punishers, where
individuals reproduce by their payoffs and replace random group members, and groups fight random other groups, the
winner's copy replacing the loser. Conflicts between groups favour the punishers that individual selection removes.
"""

# %%
# Fifty groups start with uniformly drawn numbers of punishers, with individual selection `w = 0.1` and conflicts at
# the rate `Lambda = 0.2` per group. The default blueprint shows the number of punishers of every group.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'two_level_birth_death'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=100)

# %%
# With and without conflicts between groups
# -----------------------------------------
# With conflicts the groups of punishers should take over the population, and without them defectors should gain,
# while drift fixes every group at all punishers or all defectors.

peaceful = Simulation(create_system(SYSTEM, conflict_rate=0.0), random_state=0).prepare().run(100).trajectory_
fig, (means, final) = plt.subplots(1, 2, figsize=(10, 4))
for name, trajectory in (('conflicts', simulation.trajectory_), ('no conflicts', peaceful)):
    share = [np.mean(trajectory.agents('groups', frame).values['punishers']) / 20 for frame in range(101)]
    means.plot(trajectory.times, share, label=name)
    final.hist(trajectory.agents('groups', 100).values['punishers'], bins=np.arange(22) - 0.5, alpha=0.6, label=name)
means.set_xlabel('time')
means.set_ylabel('mean fraction of punishers')
means.set_title('Group conflicts sustain punishers')
means.legend()
final.set_xlabel('punishers in a group')
final.set_ylabel('groups')
final.set_title('Groups at the end')
final.legend()
fig.tight_layout()
plt.show()
