"""
# Repressilator

Three genes in a ring, each repressing the next, cannot settle: when one protein is high it silences the next gene,
which releases the third, which silences the first. The proteins oscillate one after another, a third of a period
apart.
"""

# %%
# The model is created by name from the catalog with Elowitz and Leibler's rates and started with only the first
# protein. The observer saves every third frame with the default blueprint, whose primary view is the time series of
# the mRNA and protein counts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'repressilator'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=300)

# %%
# Deterministic and noisy oscillations
# ------------------------------------
# The differential equations should give regular oscillations of the three proteins in turn. The same rates as a jump
# process of single molecules should still oscillate, with irregular amplitudes and periods.

deterministic = simulation.trajectory_
noisy = Simulation(create(MODEL, stochastic=True), dt=0.5, random_state=0).run(until=300).trajectory_
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, trajectory, title in zip(axes, [deterministic, noisy], ['Differential equations', 'Jump process'], strict=True):
    for gene in range(3):
        ax.plot(trajectory.times, trajectory[f'gene_{gene}.p'], label=f'protein {gene}')
    ax.set_ylabel('protein count')
    ax.set_title(title)
axes[0].legend(loc='upper right')
axes[1].set_xlabel('time (min)')
spread = np.std(noisy['gene_0.p'][noisy.times > 100]) / np.mean(noisy['gene_0.p'][noisy.times > 100])
fig.suptitle(f'Proteins peak in turn; coefficient of variation with noise {spread:.2f}')
fig.tight_layout()
plt.show()
