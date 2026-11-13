"""
# Quantum Heisenberg system

One flipped spin in a ring of 12 spins one half spreads as a magnon. The space-time image shows the local
magnetisation of every site, one row per frame with the earliest at the top, with the flip leaving the middle site in a
light cone whose edges move at the largest group velocity of the chain.
"""

# %%
# The example builds `QuantumHeisenbergSystem` with the isotropic exchange `J = Δ = 1`, from one flipped spin at
# site 6. The state is a vector of `2^12 = 4096` complex amplitudes that QuTiP evolves by the Schrödinger equation.
# The observer saves every frame with the default blueprint, whose primary view is the space-time image of `S^z`.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import SectorSpec
from skcomplex.systems.physics.common.quantum import operator_measures
from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem, total_magnetization

SYSTEM = 'quantum_heisenberg'

system = QuantumHeisenbergSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(system, observers=[observer])

# %%
# Magnon and ground-state energy
# ------------------------------
# In the XX chain, `Δ = 0`, one flipped spin is a free particle that hops with amplitude `J / 2`, so its probability
# at distance `d` is `J_d(J t)²`, with Bessel functions of the first kind. The left panel compares the probability
# `1/2 - <S^z>` at three distances with this law. The right panel shows the ground-state energy per site of
# antiferromagnetic rings from exact diagonalisation in the sector of zero magnetisation. It approaches the Bethe
# ansatz value `1/4 - ln 2` of the infinite chain as the ring grows.

magnon = simulate(QuantumHeisenbergSystem(delta=0.0), until=4.0)


sizes = np.arange(4, 15, 2)
sector = SectorSpec(total_magnetization, 0.0)
energies = [
    operator_measures(QuantumHeisenbergSystem(n_sites=int(n)).prepare().system_, n_states=1, sector=sector).energies[0]
    / n
    for n in sizes
]
fig, (spread, ground) = plt.subplots(1, 2, figsize=(10, 4))
for distance in (0, 1, 3):
    spread.plot(magnon.times, 0.5 - magnon.get_values('sz')[:, 6 + distance], label=f'd = {distance}')
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
