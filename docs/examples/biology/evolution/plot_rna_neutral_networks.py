"""
# RNA neutral networks

Each RNA sequence folds into a secondary structure, and point mutations connect sequences. A neutral network is a set of
sequences of one structure that point mutations join. A walker that accepts only mutations that keep the structure
drifts far through sequence space while its phenotype stays fixed, as Raman (2021 §13.2) describes for genotype spaces.
"""

# %%
# Eight walkers of length 30 start on one sequence, and each proposes one point mutation per unit of time. The default
# blueprint shows one row per walker and one column per site, with one color per base. The rows start identical and
# come apart as the walkers drift.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system
from skcomplex.systems.biology.evolution import NeutralNetworkAnalysis, fold_sequence

SYSTEM = 'rna_neutral_networks'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = Simulation(system, observers=[observer], random_state=0).prepare().run(until=299).trajectory_

# %%
# Drift, robustness, and evolvability
# -----------------------------------
# The Hamming distance from the start grows while every sequence keeps the target fold. Wagner's genotype robustness is
# the number of neighbours of a sequence with the same fold, and its evolvability the number of other folds among its
# neighbours. Pooled over the walkers and every tenth frame, the two correlate negatively, as Wagner found.


def fold_of(sequence):
    """Return the structure of a sequence."""
    return fold_sequence(sequence)[0]


letters = trajectory.lattice('walkers.base')
sequences = np.array([[''.join(row) for row in frame] for frame in letters], dtype=object)
distances = (letters != letters[0]).sum(axis=2)
samples = sequences[::10]
robustness = NeutralNetworkAnalysis(fold_of, 'genotype-robustness').prepare(samples).run().values_.ravel()
evolvability = NeutralNetworkAnalysis(fold_of, 'genotype-evolvability').prepare(samples).run().values_.ravel()
correlation = spearmanr(robustness, evolvability).statistic

fig, (drift, trade) = plt.subplots(1, 2, figsize=(11, 4))
drift.plot(trajectory.times, distances, color='0.7', linewidth=0.8)
drift.plot(trajectory.times, distances.mean(axis=1), color='C0', label='mean of the walkers')
drift.set_xlabel('time')
drift.set_ylabel('Hamming distance from the start')
drift.set_title('The walkers drift on one fold')
drift.legend()
trade.scatter(robustness, evolvability, s=8, alpha=0.5)
trade.set_xlabel('genotype robustness')
trade.set_ylabel('genotype evolvability')
trade.set_title(f'Spearman correlation {correlation:.2f}')
fig.tight_layout()
plt.show()
