"""
# NKCS

Species climb their own NK fitness landscapes, but every gene also depends on genes of other species, so each
adaptation reshapes the landscapes of the species that depend on it. Smooth landscapes keep the ecosystem adapting
forever, rugged ones freeze it once every species sits on a peak at the same time.
"""

# %%
# The example builds `NKCSSystem` with ten species coupled to two partners each. The observer saves every frame
# with the default blueprint, whose primary view is the count of species.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import NKCSSystem

SYSTEM = 'nkcs'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(NKCSSystem(), observers=[observer], random_state=0)

# %%
# Frozen and chaotic regimes
# --------------------------
# The accepted mutations per step should die out for `k` well above `c s = 2` and keep going for smaller `k`.

fig, ax = plt.subplots(figsize=(8, 4.5))
for k in (0, 2, 8):
    run = simulate(NKCSSystem(k=k), random_state=0)
    moves = [run.get_agents('species', frame).values['moves'].sum() for frame in range(201)]
    ax.plot(np.convolve(moves, np.ones(10) / 10, mode='same'), label=f'k = {k}')
ax.set_xlabel('step')
ax.set_ylabel('accepted mutations per step')
ax.set_title('Coevolutionary activity against ruggedness')
ax.legend()
plt.show()
