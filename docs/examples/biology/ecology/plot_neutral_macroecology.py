"""
# Neutral macroecology

A hundred species, all alike, are born, die, and immigrate at random. Although no species is better than another,
their abundances spread out from the same start into a broad distribution, a few common species and many rare ones,
as in the abundances of tropical trees.
"""

# %%
# The model is created by name from the catalog with a hundred species of ten individuals each. The observer saves
# every frame with the default blueprint, whose primary view is the time series of the abundances.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'neutral_macroecology'

system = create(MODEL, species=100, n=10.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0).run(until=100)

# %%
# The negative binomial law
# -------------------------
# Without emigration the abundances settle on the negative binomial distribution with `b0 / b1` as its shape and
# `b1 / d1` as its success ratio, of mean `b0 / (d1 - b1) = 10`. Pooling the abundances of the last half of the run,
# the histogram should follow it.

abundances = simulation.trajectory_['species.n'][50:].ravel()
b0, b1, d1 = 1.0, 0.9, 1.0
counts = np.arange(abundances.max() + 1)
law = nbinom.pmf(counts, b0 / b1, 1 - b1 / d1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(abundances, bins=np.arange(abundances.max() + 2) - 0.5, density=True, alpha=0.6, label='simulated')
ax.plot(counts, law, color='black', label='negative binomial')
ax.set_xlim(0, 60)
ax.set_xlabel('abundance of a species')
ax.set_ylabel('probability')
ax.set_title(f'Mean abundance {abundances.mean():.1f}, predicted {b0 / (d1 - b1):.0f}')
ax.legend()
plt.show()
