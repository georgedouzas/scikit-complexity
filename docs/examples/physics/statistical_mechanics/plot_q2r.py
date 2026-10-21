"""
# Q2R

The Q2R automaton is a deterministic, reversible Ising dynamics: a spin flips only when its four neighbours are split
two against two, so the flip costs no energy, and the two sublattices of the checkerboard take turns. The energy is
conserved exactly, so the automaton samples the microcanonical ensemble of the Ising model. Started from a random
lattice with 10% of the spins up, which has the energy of an ordered Ising magnet, minority spins wander through the
majority as ragged clusters. The view shows the spins of a 128 x 128 torus.
"""

# %%
# The model is created by name from the catalog with an initial fraction `up = 0.1` of spins up. One step updates one
# sublattice. The observer saves every step with the default blueprint, whose primary view is the image of the spins.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'q2r'

system = create(MODEL, shape=(128, 128), up=0.1, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# Microcanonical Ising magnetization
# ----------------------------------
# The initial fraction of up spins fixes the conserved energy per site `u = -<s_i s_j>` summed over the two bonds of a
# site. If the automaton is ergodic, its long-time magnetization at that energy should match the spontaneous
# magnetization of the Ising model at the temperature whose Onsager energy is `u`, zero above the critical energy
# `u_c = -sqrt 2`, with the rounding of a finite lattice near `u_c`.


def energy_and_magnetization(up):
    """Return the energy per site and the late mean absolute magnetization of a run of the automaton."""
    run = Simulation(create(MODEL, shape=(64, 64), up=up, random_state=1), dt=1).run(until=3000)
    spins = run.trajectory_['spins.spin'].reshape(-1, 64, 64)
    bonds = spins * np.roll(spins, 1, axis=1) + spins * np.roll(spins, 1, axis=2)
    return -bonds.mean(axis=(1, 2))[-1], np.abs(spins[1000:].mean(axis=(1, 2))).mean()


measured = np.array([energy_and_magnetization(up) for up in np.linspace(0.02, 0.3, 12)])
temperatures = np.linspace(0.8, 2 / np.log(1 + np.sqrt(2)) - 1e-6, 300)
coupling = 1 / temperatures
modulus = 2 * np.sinh(2 * coupling) / np.cosh(2 * coupling) ** 2
onsager_energy = -(1 + 2 / np.pi * (2 * np.tanh(2 * coupling) ** 2 - 1) * ellipk(modulus**2)) / np.tanh(2 * coupling)
onsager_magnetization = np.clip(1 - np.sinh(2 * coupling) ** -4, 0, None) ** 0.125
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(measured[:, 0], measured[:, 1], 'o', label='Q2R, 64 x 64')
ax.plot(onsager_energy, onsager_magnetization, color='black', label='Onsager m(u)')
ax.plot([-np.sqrt(2), 0], [0, 0], color='black')
ax.axvline(-np.sqrt(2), color='gray', linestyle='--', label='u_c = -sqrt 2')
ax.set_xlabel('energy per site u')
ax.set_ylabel('mean |m|')
ax.set_title('The automaton orders below the critical energy')
ax.legend()
fig.tight_layout()
plt.show()
