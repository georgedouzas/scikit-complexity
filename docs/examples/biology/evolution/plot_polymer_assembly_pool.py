"""
# Polymer assembly pool

A pool grows from one monomer by joining pairs of its polymers. Random joining fills it with many short polymers, while
reusing the newest polymer, a simple model of selection, builds ever longer chains of high assembly index.
"""

# %%
# The system is created by name from the catalog with undirected joining, and the observer saves every step with the
# default blueprint, whose primary view is the count of polymers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'polymer_assembly_pool'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=200)

# %%
# Assembly index of the longest polymer
# -------------------------------------
# The assembly index of a polymer is about `log2` of its length. Averaged over runs, directed joining should reach far
# higher indices than undirected joining.

fig, ax = plt.subplots(figsize=(8, 4.5))
for mode in ('undirected', 'directed'):
    runs = []
    for seed in range(10):
        run = Simulation(create_system(SYSTEM, mode=mode), random_state=seed).prepare().run(until=100).trajectory_
        runs.append([np.log2(run.agents('pool', frame).values['length'].max()) for frame in range(101)])
    ax.plot(np.mean(runs, axis=0), label=mode)
ax.set_xlabel('step')
ax.set_ylabel('mean largest assembly index')
ax.set_title('Selection builds more complex polymers')
ax.legend()
plt.show()
