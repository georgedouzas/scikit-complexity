"""
# Bak-Sneppen

The least fit species of a ring and its two neighbours are replaced, one update at a time, by species of random
fitness. The fitnesses self-organise above a critical value close to 0.667, and the replacements come in avalanches of
every size. The view shows the space-time history of the fitnesses of the ring, the latest step at the bottom.
"""

# %%
# The model is created by name from the catalog on a ring of 200 species with 200 updates per step. The observer
# saves every tenth step with the default blueprint, whose primary view is the space-time image of the fitnesses.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bak_sneppen'

system = create(MODEL, size=200)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=10, observers=[observer], random_state=0).run(until=2000)

# %%
# The gap and the distribution of fitnesses
# -----------------------------------------
# Every update is an `extinction` event with the minimum fitness it replaced. The gap, the largest minimum so far,
# should rise in steps towards the critical value `f_c = 0.667` of the one-dimensional model, and at the end the
# fitnesses should be spread uniformly above `f_c`, with few species below it.

minima = np.asarray([event.value for event in simulation.trajectory_.events])
gap = np.maximum.accumulate(minima)
fig, (rise, spread) = plt.subplots(1, 2, figsize=(10, 4))
rise.semilogx(np.arange(1, minima.size + 1), gap, label='gap')
rise.axhline(0.667, color='black', linestyle='--', label='f_c = 0.667')
rise.set_xlabel('update')
rise.set_ylabel('largest minimum fitness so far')
rise.set_title('The gap rises towards f_c')
rise.legend()
spread.hist(simulation.state_, bins=20, range=(0, 1), density=True)
spread.axvline(0.667, color='black', linestyle='--', label='f_c = 0.667')
spread.set_xlabel('fitness')
spread.set_ylabel('density')
spread.set_title('Fitnesses gather above f_c')
spread.legend()
fig.tight_layout()
plt.show()
