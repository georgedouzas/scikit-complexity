"""
# Neutral biodiversity

The number of species of a neutral community of 200 individuals, all of one species at the start, rises as new
species arise and then fluctuates around a stationary value, while every single species appears and dies out by
drift.
"""

# %%
# The model is created by name from the catalog with the speciation probability `nu = 0.01` per birth. One unit of
# time holds 200 deaths, each replaced by the offspring of another individual. The observer saves every fourth frame
# with the default blueprint, whose primary view is the time series of the number of species.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'neutral_biodiversity'
SIZE, NU = 200, 0.01

system = create(MODEL, individuals=SIZE, nu=NU, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=1000)

# %%
# Number and abundances of species
# --------------------------------
# With `theta = (N - 1) nu / (1 - nu)`, the mean number of species is `S_N = sum_n theta / (theta + n)` and the mean
# number of species of `k` individuals is `phi_k = theta / k prod_j (N - j + 1) / (N - j + theta)`. After the first
# 200 units of time, the number of species should fluctuate around `S_N`, and the abundances counted at every frame,
# summed over bins of doubling width, should follow `phi_k`.

theta = (SIZE - 1) * NU / (1 - NU)
expected_richness = np.sum(theta / (theta + np.arange(SIZE)))
abundances = np.arange(1, SIZE + 1)
phi = theta / abundances * np.cumprod((SIZE - abundances + 1) / (SIZE - abundances + theta))
times = simulation.trajectory_.times
counts = np.zeros(SIZE + 1)
for frame in np.flatnonzero(times >= 200):
    _, sizes = np.unique(simulation.trajectory_.agents('community', int(frame)).values['species'], return_counts=True)
    counts += np.bincount(sizes, minlength=SIZE + 1)
counts /= np.sum(times >= 200)
bins = list(itertools.pairwise(2 ** np.arange(9)))

fig, (richness, distribution) = plt.subplots(1, 2, figsize=(10, 4))
richness.plot(times, simulation.trajectory_['richness'], label='simulated')
richness.axhline(expected_richness, color='black', linestyle='--', label='S_N')
richness.set_xlabel('time')
richness.set_ylabel('number of species')
richness.set_title('Biodiversity settles around S_N')
richness.legend()
binned = [counts[low:high].sum() for low, high in bins]
predicted = [phi[low - 1 : high - 1].sum() for low, high in bins]
distribution.bar(np.arange(len(binned)), binned, label='simulated')
distribution.plot(np.arange(len(predicted)), predicted, 'o', color='black', label='phi_k')
distribution.set_xticks(np.arange(len(binned)), [f'{low}-{high - 1}' for low, high in bins])
distribution.set_xlabel('individuals k of a species')
distribution.set_ylabel('mean number of species')
distribution.set_title('Species abundance distribution')
distribution.legend()
fig.tight_layout()
plt.show()
