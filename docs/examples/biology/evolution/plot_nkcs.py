"""
# NKCS

Species climb their own NK fitness landscapes, but every gene also depends on genes of other species, so each
adaptation reshapes the landscapes of the species that depend on it. Smooth landscapes keep the ecosystem adapting
forever, rugged ones freeze it once every species sits on a peak at the same time.
"""

# %%
# The system is created by name from the catalog with ten species coupled to two partners each. The observer saves
# every frame with the default blueprint, whose primary view is the count of species.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'nkcs'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=200)

# %%
# Frozen and chaotic regimes
# --------------------------
# The accepted mutations per step should die out for `k` well above `c s = 2` and keep going for smaller `k`.

fig, ax = plt.subplots(figsize=(8, 4.5))
for k in (0, 2, 8):
    run = Simulation(create_system(SYSTEM, k=k), random_state=0).prepare().run(until=200).trajectory_
    moves = [run.agents('species', frame).values['moves'].sum() for frame in range(201)]
    ax.plot(np.convolve(moves, np.ones(10) / 10, mode='same'), label=f'k = {k}')
ax.set_xlabel('step')
ax.set_ylabel('accepted mutations per step')
ax.set_title('Coevolutionary activity against ruggedness')
ax.legend()
plt.show()
