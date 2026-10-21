"""
# XY model

Planar spins on a square lattice prefer to point like their neighbours. Quenched from random angles to a low
temperature, they align over large smooth domains, but the alignment winds around isolated points, vortices and
antivortices, which the spins cannot unwind locally and which annihilate only in pairs. Above the
Kosterlitz-Thouless temperature, near 0.89, free vortices proliferate.
"""

# %%
# A 64 x 64 lattice starts from random angles at the temperature 0.3. The default blueprint colors the angle of every
# spin around a cyclic palette, so a vortex is a point where all the colors meet.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.base import wrap_angle
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'xy'

system = create(MODEL, size=64, temperature=0.3, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=200)

# %%
# Vortices and energy against temperature
# ---------------------------------------
# A vortex is a plaquette around which the angle differences, wrapped into `(-π, π]`, add up to a full turn. Their
# density should be small below the Kosterlitz-Thouless point and rise steeply above it. At low temperature the energy
# per spin should follow the spin-wave line `-2 J + T / 2`, one half of `T` per quadratic mode, and rise above it once
# vortices appear.


def measure(temperature):
    """Return the vortex density and the energy per spin over the last 100 of 400 sweeps from random angles."""
    run = Simulation(create(MODEL, size=32, temperature=temperature, random_state=1), random_state=1)
    theta = np.asarray(run.run(until=400).trajectory_.lattice('spins.theta'))[-100:]
    right, down = np.roll(theta, -1, axis=-1), np.roll(theta, -1, axis=-2)
    energy = -(np.cos(theta - right) + np.cos(theta - down)).mean()
    winding = wrap_angle(right - theta) + wrap_angle(np.roll(right, -1, -2) - right)
    winding += wrap_angle(down - np.roll(right, -1, -2)) + wrap_angle(theta - down)
    return np.mean(np.abs(winding) > np.pi), energy


temperatures = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
density, energy = np.array([measure(temperature) for temperature in temperatures]).T
fig, (vortices, energies) = plt.subplots(1, 2, figsize=(11, 4))
vortices.plot(temperatures, density, 'o-')
vortices.axvline(0.89, color='grey', linestyle=':', label='Kosterlitz-Thouless')
vortices.set_xlabel('temperature')
vortices.set_ylabel('vortices per plaquette')
vortices.set_title('Free vortices appear near the transition')
vortices.legend()
energies.plot(temperatures, energy, 'o', label='simulated')
energies.plot(temperatures, -2 + temperatures / 2, '--', color='black', label='spin waves, -2J + T/2')
energies.set_xlabel('temperature')
energies.set_ylabel('energy per spin')
energies.set_title('Spin waves at low temperature')
energies.legend()
fig.tight_layout()
plt.show()
