"""
# Origin-fixation walk

The genome of a monomorphic population of 100 individuals, one column per locus and one row per frame from the oldest
at the top, climbs a landscape of random fields as mutants fix one at a time: from the genome of all `-1`, most loci
switch to their fitter allele, while loci with weak fields keep flipping between the two.
"""

# %%
# The model is created by name from the catalog with 96 loci, random fields of standard deviation 0.01, no couplings,
# and the per-locus mutation rate 0.01, so mutants of a locus arise at the rate 1. The observer saves every frame with
# the default blueprint, whose primary view is the space-time image of the alleles.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'origin_fixation_walk'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=10.0,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=640)

# %%
# Climbing and the stationary law
# -------------------------------
# Under strong selection and weak mutation the genome visits every genome `sigma` with the probability
# `Q(sigma) ~ f(sigma)^(N - 1)`, the Boltzmann law of the energy `-ln f` at the temperature `1 / (N - 1)`. Without
# couplings the landscape has one peak, of fitness `1 + sum_i |h_i|`. Expanding `ln f` around the mean fitness `f*`
# makes the loci independent, with `<s_i> = tanh((N - 1) h_i / f*)`, so the fitness of the run should climb and then
# fluctuate around the root of `f* = 1 + sum_i h_i tanh((N - 1) h_i / f*)`, below the peak since the drift of a
# population of 100 keeps loci with weak fields mixed. A long run of four loci with couplings, a rugged landscape,
# should spend on each of its 16 genomes the share of time that this law predicts.

fields = system.elements[0].values['h']
stationary = 1.0
for _ in range(100):
    stationary = 1 + np.sum(fields * np.tanh(99 * fields / stationary))
small = create(MODEL, loci=4, N=10.0, mu=1.0, field=0.1, coupling=0.3, random_state=1)
alleles = (
    Simulation(small, dt=1.0, random_state=0, observables=describe(MODEL).observables_of(small))
    .run(until=5000)
    .trajectory_['genome.s'][100:]
)
visited = np.bincount(((alleles > 0) * 2 ** np.arange(4)).sum(axis=1).astype(int), minlength=16)
genomes = np.array([[2 * bit - 1 for bit in reversed(bits)] for bits in itertools.product((0, 1), repeat=4)])
couplings = small.interactions[0].couplings
fitness = 1.0 + genomes @ small.elements[0].values['h'] + np.einsum('gi,ij,gj->g', genomes, couplings, genomes) / 2

fig, (climb, law) = plt.subplots(1, 2, figsize=(10, 4))
climb.plot(simulation.trajectory_.times, simulation.trajectory_['fitness'], label='simulated')
climb.axhline(1 + np.abs(fields).sum(), color='black', linestyle='--', label='peak of the landscape')
climb.axhline(stationary, color='black', linestyle=':', label='stationary f*')
climb.set_xlabel('time')
climb.set_ylabel('fitness f')
climb.set_title('The population climbs the landscape')
climb.legend()
law.bar(np.arange(16), visited / visited.sum(), label='share of time')
law.plot(np.arange(16), fitness**9 / np.sum(fitness**9), 'o', color='black', label='f^(N - 1) / Z')
law.set_xlabel('genome')
law.set_ylabel('probability')
law.set_title('Stationary law of four loci, N = 10')
law.legend()
fig.tight_layout()
plt.show()
