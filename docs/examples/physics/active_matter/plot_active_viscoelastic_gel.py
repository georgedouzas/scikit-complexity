"""
# Active viscoelastic gel

Contractile cells pull on the gel around them with a stress that grows with their density, and the gel drags them along.
Above a threshold of contractility, this feedback beats their diffusion, and a nearly uniform density concentrates into
peaks. The gel has no inertia, so its velocity solves a force balance at every step.
"""

# %%
# The system is built from its class on a periodic line of length 10 in 128 sites, around the density 1,
# with `c Ta'(c) = 1` above `gamma D = 0.1`. The catalog entry solves the force balance with `inverse_helmholtz` and
# streams the velocity as an observable, and the default blueprint shows the history of the density.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import ActiveViscoelasticGelSystem

SYSTEM = 'active_viscoelastic_gel'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ActiveViscoelasticGelSystem(), observers=[observer], random_state=0)

# %%
# Concentration and force balance
# -------------------------------
# The largest density should rise far above 1 while the total stays fixed, and the cells should flow toward the peak.
# Applying `gamma - eta d_xx` to the streamed velocity should return the gradient of the stress to rounding.

density = trajectory.get_lattice('gel.c')
displacement = trajectory.get_lattice('gel.u')
velocity = np.asarray(trajectory.get_values('v'))
spacing = 10 / 128
centers = (np.arange(128) + 0.5) * spacing


def central(field):
    """Return the central difference of a field along the line."""
    return (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2 * spacing)


stress = 0.1 * central(displacement) + 4.0 * density / (density + 1.0)
second = (np.roll(velocity, -1, axis=-1) - 2 * velocity + np.roll(velocity, 1, axis=-1)) / spacing**2
residual = np.abs(velocity - second - central(stress)).max()

fig, (growth, profile) = plt.subplots(1, 2, figsize=(10, 4))
growth.semilogy(trajectory.times, np.ptp(density, axis=1))
growth.set_xlabel('time')
growth.set_ylabel('range of the density')
growth.set_title(f'Force balance holds within {residual:.0e}')
profile.plot(centers, density[-1], label='density')
profile.plot(centers, velocity[-1], label='velocity')
profile.set_xlabel('position')
profile.set_title('The gel flows toward the peak')
profile.legend()
fig.tight_layout()
plt.show()
