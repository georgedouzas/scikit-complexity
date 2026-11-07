"""
# NK landscape

Fifty climbers start from random genomes on Kauffman's NK landscape and accept every single mutation that raises their
fitness. They climb quickly and stop on local peaks. The more the genes interact, the more rugged the landscape, and
the more different peaks trap the climbers.
"""

# %%
# Fifty climbers have genomes of ten genes, each interacting with two others. The default blueprint shows their
# mean fitness rising quickly and levelling off as they stop on local peaks.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import NKLandscapeSystem

SYSTEM = 'nk_landscape'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(NKLandscapeSystem(climbers=50, n=10, k=2, seed=0), dt=1, observers=[observer], random_state=0)

# %%
# Ruggedness
# ----------
# With `k = 0` the landscape has a single peak and every climber should reach it, while with larger `k` the climbers
# should end on ever more distinct peaks, nearly one per climber on the random landscape `k = n - 1`.


def final_climbers(k):
    """Return the final genomes and fitnesses of a run with `k` interactions."""
    trajectory = simulate(NKLandscapeSystem(climbers=50, n=10, k=k, seed=0), until=150, dt=1, random_state=0)
    climbers = trajectory.agents('climbers', len(trajectory.times) - 1)
    return climbers.values['genome'], climbers.values['fitness']


ks = [0, 2, 4, 6, 9]
peaks = [len({str(genome) for genome in final_climbers(k)[0]}) for k in ks]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, peaks, 'o-')
ax.set_xlabel('interactions k')
ax.set_ylabel('distinct final genomes of 50 climbers')
ax.set_title(f'One peak at k = 0, {peaks[-1]} peaks at k = 9')
plt.show()
