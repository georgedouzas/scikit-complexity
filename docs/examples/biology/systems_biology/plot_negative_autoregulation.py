"""
# Negative autoregulation

A gene whose mRNA lives briefly makes its protein in bursts of about ten molecules, so the protein count jumps up
and decays in a jagged sawtooth. When the protein also represses its own gene, the count stays lower and fluctuates
less.
"""

# %%
# The model runs as a jump process of single transcriptions, translations, and decays. The observer saves every frame
# with the default blueprint, whose primary view is the time series of the mRNA and protein counts.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import fano_factor
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'negative_autoregulation'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0).run(until=200)

# %%
# Feedback reduces the noise
# --------------------------
# Bursty expression makes the Fano factor, the variance over the mean, about `1 + b / delta = 11` without feedback,
# far above the 1 of a Poisson count. With feedback both the mean and the Fano factor of the protein should drop.

fig, ax = plt.subplots(figsize=(8, 4))
for label, K in [('with feedback', 50.0), ('without feedback', 1e9)]:
    trajectory = Simulation(create(MODEL, K=K), dt=1.0, random_state=1).run(until=1000).trajectory_
    protein = trajectory['gene.p'][trajectory.times > 100]
    ax.plot(
        trajectory.times,
        trajectory['gene.p'],
        label=f'{label}: mean {protein.mean():.0f}, Fano {fano_factor(protein):.1f}',
    )
ax.set_xlabel('time')
ax.set_ylabel('protein count')
ax.set_title('Negative autoregulation lowers the mean and the noise')
ax.legend()
plt.show()
