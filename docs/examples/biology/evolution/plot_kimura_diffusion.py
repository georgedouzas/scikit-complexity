"""
# Kimura diffusion

Allele frequencies of many replicate populations drift at random and are pushed by selection, until each allele is
lost or fixed. The share of replicates in which the allele fixes is Kimura's fixation probability.
"""

# %%
# The example builds `KimuraDiffusionSystem` with 200 replicates starting at the frequency 0.1 with the scaled selection
# `N s = 2`, and integrates it by Euler-Maruyama steps of 0.001. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the frequencies.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import KimuraDiffusionSystem

SYSTEM = 'kimura_diffusion'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = EulerMaruyamaSolver(dt=0.001)
trajectory = simulate(KimuraDiffusionSystem(replicates=200), solver=solver, observers=[observer], random_state=0)

# %%
# Fixation probability
# --------------------
# The share of fixed replicates should follow `(1 - exp(-N s x)) / (1 - exp(-N s))` as a function of the initial
# frequency `x`.

starts = np.linspace(0.1, 0.9, 5)
fixed = []
for start in starts:
    run = simulate(KimuraDiffusionSystem(x=start, replicates=500), until=10, dt=1.0, solver=solver, random_state=1)
    fixed.append((run.get_values('alleles.x')[-1] > 0.999).mean())
grid = np.linspace(0, 1, 100)
fig, (paths, probability) = plt.subplots(1, 2, figsize=(11, 4.5))
paths.plot(trajectory.times, trajectory.get_values('alleles.x')[:, :30], lw=0.6)
paths.set_xlabel('time (N generations)')
paths.set_ylabel('allele frequency')
paths.set_title('Drift to loss or fixation')
probability.plot(starts, fixed, 'o', label='replicates')
probability.plot(grid, (1 - np.exp(-2 * grid)) / (1 - np.exp(-2)), label='Kimura')
probability.set_xlabel('initial frequency')
probability.set_ylabel('fixation probability')
probability.set_title('N s = 2')
probability.legend()
fig.tight_layout()
plt.show()
