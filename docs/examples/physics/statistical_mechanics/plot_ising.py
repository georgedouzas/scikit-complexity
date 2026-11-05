"""
# Ising

Quenched from random spins to a temperature below the critical one `T_c = 2 / ln(1 + sqrt 2) = 2.27 J`, the
two-dimensional Ising model coarsens: domains of up and down spins grow and their walls straighten until one
orientation wins. The view shows the spins of a 256 x 256 torus 100 sweeps after the quench, purple down and yellow up.
"""

# %%
# The model is created by name from the catalog at `T = 2.0 J`, below `T_c`. One unit of time is one sweep of
# checkerboard Metropolis updates. The observer saves every frame with the default blueprint, whose primary view is
# the image of the spins.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import InformationAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ising'

system = create(MODEL, size=256, temperature=2.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).prepare().run(until=100)

# %%
# Spontaneous magnetization against Onsager's solution
# ----------------------------------------------------
# Onsager and Yang solved the square lattice exactly: the spontaneous magnetization is
# `m(T) = (1 - sinh(2 J / T)^-4)^(1/8)` below `T_c` and zero above it. Started from all spins up, a 48 x 48 lattice
# should keep its mean absolute magnetization on this curve below `T_c`, round off near `T_c` where the correlation
# length reaches the size of the lattice, and fall towards zero above it.

temperatures = np.linspace(1.5, 3.5, 17)
measured = []
for temperature in temperatures:
    system_t = create(MODEL, size=48, temperature=temperature, initial='up')
    run = Simulation(system_t, dt=1, random_state=1).prepare().run(until=1500)
    measured.append(np.abs(run.trajectory_.values('spins.spin')[300:].mean(axis=1)).mean())
grid = np.linspace(1.5, 2 / np.log(1 + np.sqrt(2)), 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temperatures, measured, 'o', label='simulation, 48 x 48')
ax.plot(grid, np.clip(1 - np.sinh(2 / grid) ** -4, 0, None) ** 0.125, color='black', label='Onsager, infinite lattice')
ax.axvline(2 / np.log(1 + np.sqrt(2)), color='gray', linestyle='--', label='T_c')
ax.set_xlabel('temperature T / J')
ax.set_ylabel('mean |m|')
ax.set_title('Spontaneous magnetization of the square-lattice Ising model')
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Redundancy peaks at the critical point
# --------------------------------------
# The O-information of a block of spins, the kind `'o-information'` of `InformationAnalysis`, is the total
# correlation minus the dual total correlation: positive when the spins share the same information, redundancy, and
# negative when information lies only in their joint state, synergy. Ising spins should be redundant at every
# temperature, and their redundancy should peak at `T_c`, where correlations span the lattice (Scagliarini et al. 2022).
# Every 2 x 2 block of a 48 x 48 lattice, sampled every 10 sweeps after a burn-in of 300, is one sample of four binary
# variables.

information = []
for temperature in temperatures:
    system_t = create(MODEL, size=48, temperature=temperature, initial='up')
    run = Simulation(system_t, dt=1, random_state=1).prepare().run(until=600)
    frames = np.asarray(run.trajectory_.values('spins.spin')[300::10]).reshape(-1, 48, 48)
    blocks = frames.reshape(-1, 24, 2, 24, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 4)
    analysis = InformationAnalysis('o-information').prepare((blocks > 0).astype(int)).run()
    information.append(analysis.information_)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temperatures, information, 'o-')
ax.axvline(2 / np.log(1 + np.sqrt(2)), color='gray', linestyle='--', label='T_c')
ax.set_xlabel('temperature T / J')
ax.set_ylabel('O-information of a 2 x 2 block (bits)')
ax.set_title('Redundancy of neighbouring spins')
ax.legend()
fig.tight_layout()
plt.show()
