"""
# Brownian motion of Janus, ellipsoidal, and chiral particles in three dimensions

Colloids that are not plain spheres reveal their rotation: a Janus particle shows its orientation, an ellipsoid
diffuses faster along its long axis than across it, and a spiral couples its translation along the axis to its spin.
"""

# %%
# Two hundred Janus particles start at the origin with random axes. The default blueprint shows the positions, the
# axes, and the spins.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import AnisotropicBrownianParticlesSystem

SYSTEM = 'anisotropic_brownian_particles'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = simulate(AnisotropicBrownianParticlesSystem(), until=10, observers=[observer], random_state=0)

# %%
# Rotation and anisotropy
# -----------------------
# The axes should decorrelate as `exp(-2 D_r t)`, and for an ellipsoid with `D_par = 1.5` and `D_perp = 0.5` the
# displacements in the frame of the initial axis should be elongated along it at short times, and the gap should
# shrink as the axis turns, towards the mean coefficient `(D_par + 2 D_perp) / 3`.

trajectory = simulate(AnisotropicBrownianParticlesSystem(D_par=1.5, D_perp=0.5, D_r=0.5), until=10, random_state=0)
times = np.asarray(trajectory.times)
axes = np.stack([np.asarray(trajectory.get_values(f'particles.u{name}')) for name in 'xyz'], axis=-1)
positions = np.stack([np.asarray(trajectory.get_values(f'particles.{name}')) for name in 'xyz'], axis=-1)
along = (positions * axes[0]).sum(axis=-1)
across = (positions**2).sum(axis=-1) - along**2
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.semilogy(times, (axes * axes[0]).sum(axis=-1).mean(axis=1), label='simulation')
left.semilogy(times, np.exp(-2 * 0.5 * times), '--', color='black', label='exp(-2 D_r t)')
left.set_ylim(1e-2, 1.1)
left.set_xlabel('time')
left.set_ylabel('<u(t) . u(0)>')
left.legend()
right.plot(times[1:], (along**2).mean(axis=1)[1:] / (2 * times[1:]), label='along the initial axis')
right.plot(times[1:], across.mean(axis=1)[1:] / (4 * times[1:]), label='across it')
right.set_xlabel('time')
right.set_ylabel('apparent diffusion coefficient')
right.legend()
plt.show()
