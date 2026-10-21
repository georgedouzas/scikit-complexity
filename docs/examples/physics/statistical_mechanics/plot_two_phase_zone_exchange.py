"""
# Zone exchange in a two-phase apparatus

A small zone of an apparatus, a tenth of its volume, exchanges particles with the rest: each of the 100 particles
enters the zone at rate `k p` and leaves it at rate `k (1 - p)`. Started empty, the zones fill within a few exchange
times and then fluctuate around ten particles, each zone independently. The view shows the numbers of particles in
ten zones rising from zero and then jumping up and down around the mean.
"""

# %%
# The model is created by name from the catalog with ten zones, `Na = 100` particles, volume fraction `p = 0.1`, and
# exchange rate `k = 1`. The observer saves every frame of 0.1 with the default blueprint, whose primary view is the
# time series of the numbers in the zones.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'two_phase_zone_exchange'

system = create(MODEL, n_zones=10)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer], random_state=0).run(until=20)

# %%
# Binomial occupation
# -------------------
# Every particle sits in the zone independently with probability `p` once the exchange has relaxed, after a few
# times `1 / k`, so the number of particles in a zone is binomial with mean `Na p = 10` and variance
# `Na p (1 - p) = 9`. Pooled over 2000 zones and the second half of the run, the histogram should match this
# distribution.

run = Simulation(create(MODEL, n_zones=2000), dt=1.0, random_state=1).run(until=20)
numbers = run.trajectory_['zones.N'][10:].ravel()
counts = np.arange(numbers.max() + 2)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(numbers, bins=counts - 0.5, density=True, alpha=0.5, label=f'simulation, variance {numbers.var():.2f}')
ax.plot(counts, stats.binom.pmf(counts, 100, 0.1), 'o-', color='black', label='binomial(100, 0.1), variance 9')
ax.set_xlabel('particles in the zone')
ax.set_ylabel('probability')
ax.set_title('The occupation of a zone is binomial')
ax.legend()
fig.tight_layout()
plt.show()
