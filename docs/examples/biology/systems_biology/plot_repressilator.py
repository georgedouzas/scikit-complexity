"""
# Repressilator

Three genes in a ring, each repressing the next, cannot settle: when one protein is high it silences the next gene,
which releases the third, which silences the first. The proteins oscillate one after another, a third of a period
apart.
"""

# %%
# The system is built from its class with Elowitz and Leibler's rates and started with only the first protein. The
# observer saves every third frame with the default blueprint, whose primary view is the time series of the mRNA and
# protein counts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import RepressilatorSystem

SYSTEM = 'repressilator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
trajectory = simulate(RepressilatorSystem(), observers=[observer], until=300)

# %%
# Deterministic and noisy oscillations
# ------------------------------------
# The differential equations should give regular oscillations of the three proteins in turn. The same rates as a jump
# process of single molecules should still oscillate, with irregular amplitudes and periods.

deterministic = trajectory
noisy = simulate(RepressilatorSystem(stochastic=True), random_state=0, until=300)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, trajectory, title in zip(axes, [deterministic, noisy], ['Differential equations', 'Jump process'], strict=True):
    for gene in range(3):
        ax.plot(trajectory.times, trajectory.get_values(f'gene_{gene}.p'), label=f'protein {gene}')
    ax.set_ylabel('protein count')
    ax.set_title(title)
axes[0].legend(loc='upper right')
axes[1].set_xlabel('time (min)')
spread = np.std(noisy.get_values('gene_0.p')[noisy.times > 100]) / np.mean(
    noisy.get_values('gene_0.p')[noisy.times > 100],
)
fig.suptitle(f'Proteins peak in turn; coefficient of variation with noise {spread:.2f}')
fig.tight_layout()
plt.show()
