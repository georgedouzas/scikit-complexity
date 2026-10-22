"""
# Quantum Heisenberg model

One flipped spin in a ring of 12 spins one half spreads as a magnon. The space-time image shows the local
magnetisation of every site, one row per frame with the earliest at the top, with the flip leaving the middle site in a
light cone whose edges move at the largest group velocity of the chain.
"""

# %%
# The model is created by name from the catalog with the isotropic exchange `J = Δ = 1`, from one flipped spin at
# site 6. The state is a vector of `2^12 = 4096` complex amplitudes that QuTiP evolves by the Schrödinger equation.
# The observer saves every frame with the default blueprint, whose primary view is the space-time image of `S^z`.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from skcomplex.core.analysis import spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe, run

MODEL = 'quantum_heisenberg'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.05, observers=[observer], observables=describe(MODEL).observables_of(system)).run(
    until=8.0,
)

# %%
# Magnon and ground-state energy
# ------------------------------
# In the XX chain, `Δ = 0`, one flipped spin is a free particle that hops with amplitude `J / 2`, so its probability
# at distance `d` is `J_d(J t)²`, with Bessel functions of the first kind. The left panel compares the probability
# `1/2 - <S^z>` at three distances with this law. The right panel shows the ground-state energy per site of
# antiferromagnetic rings from exact diagonalisation in the sector of zero magnetisation. It approaches the Bethe
# ansatz value `1/4 - ln 2` of the infinite chain as the ring grows.

magnon = run(MODEL, delta=0.0, settings={'until': 4.0, 'dt': 0.05}).trajectory_


def magnetization(s):
    """Return the total magnetisation of the chain."""
    return sum((member.z for member in s.chain), start=0 * s.chain[0].z)


sizes = np.arange(4, 15, 2)
energies = [spectrum(create(MODEL, n_sites=int(n)), k=1, sector=(magnetization, 0.0)).energies[0] / n for n in sizes]
fig, (spread, ground) = plt.subplots(1, 2, figsize=(10, 4))
for distance in (0, 1, 3):
    spread.plot(magnon.times, 0.5 - magnon['sz'][:, 6 + distance], label=f'd = {distance}')
    spread.plot(magnon.times, jv(distance, magnon.times) ** 2, color='black', linestyle=':', linewidth=1)
spread.set_xlabel('time')
spread.set_ylabel('probability of the flip')
spread.set_title('Magnon of the XX ring against J_d(t)²')
spread.legend()
ground.plot(sizes, energies, 'o-', label='exact diagonalisation')
ground.axhline(0.25 - np.log(2), color='black', linestyle='--', label='1/4 - ln 2')
ground.set_xlabel('number of sites')
ground.set_ylabel('ground-state energy per site')
ground.set_title('Antiferromagnetic rings approach the Bethe ansatz')
ground.legend()
fig.tight_layout()
plt.show()
