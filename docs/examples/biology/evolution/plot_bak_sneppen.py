"""
# Bak-Sneppen

The least fit species of a ring and its two neighbours are replaced, one extinction per step, by species of random
fitness. The fitnesses self-organise above a critical value close to 0.667, and the replacements come in avalanches of
every size. The view shows the space-time history of the fitnesses of the ring, the latest step at the bottom.
"""

# %%
# The example builds `BakSneppenSystem` on a ring of 200 species and runs 400000 extinctions. The observer saves every
# 2000th step with the default blueprint, whose primary view is the space-time image of the fitnesses.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import BakSneppenSystem

SYSTEM = 'bak_sneppen'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(BakSneppenSystem(size=200), until=400_000, dt=2_000, observers=[observer], random_state=0)

# %%
# The gap and the distribution of fitnesses
# -----------------------------------------
# The smallest fitness of every saved frame is the fitness the next extinction replaces. The gap, the largest of these
# minima so far, should rise towards the critical value `f_c = 0.667` of the one-dimensional system, and at the end the
# fitnesses should be spread uniformly above `f_c`, with few species below it.

fitnesses = trajectory.get_values('species.fitness')
minima = fitnesses.min(axis=1)
gap = np.maximum.accumulate(minima)
fig, (rise, spread) = plt.subplots(1, 2, figsize=(10, 4))
rise.semilogx(trajectory.times[1:], gap[1:], label='gap')
rise.axhline(0.667, color='black', linestyle='--', label='f_c = 0.667')
rise.set_xlabel('extinctions')
rise.set_ylabel('largest minimum fitness so far')
rise.set_title('The gap rises towards f_c')
rise.legend()
spread.hist(fitnesses[-1], bins=20, range=(0, 1), density=True)
spread.axvline(0.667, color='black', linestyle='--', label='f_c = 0.667')
spread.set_xlabel('fitness')
spread.set_ylabel('density')
spread.set_title('Fitnesses gather above f_c')
spread.legend()
fig.tight_layout()
plt.show()
