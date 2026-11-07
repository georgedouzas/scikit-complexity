"""
# Quasi-geostrophic double gyre

A steady wind over a closed basin on the beta plane spins up two gyres of opposite sign. In the interior the Sverdrup
balance sets the flow, and the return flow closes in narrow boundary layers along the western wall, the viscous system
of currents such as the Gulf Stream. The state is the potential vorticity, inverted into a streamfunction at each step.
"""

# %%
# The system is built from its class on a square basin of 64 by 64 sites with walls, in its viscous steady
# regime, and integrated with RK4 for 400 time units. The catalog entry streams the streamfunction as the observable
# `psi`, and the default blueprint shows it as an image.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.oceanography import DoubleGyreSystem

SYSTEM = 'double_gyre'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=50)
trajectory = simulate(DoubleGyreSystem(), observers=[observer], random_state=0)

# %%
# Western intensification
# -----------------------
# The streamfunction should be positive in the southern half and negative in the northern one, with its extrema near the
# western wall. Along the central latitude of each gyre, it should follow the Sverdrup balance `beta psi_x = curl(tau)`
# away from the boundary layer.

psi = np.asarray(trajectory.values('psi'))[-1].reshape(64, 64)
positions = (np.arange(64) + 0.5) / 64
sverdrup = 1e-5 * 2 * np.pi * np.sin(2 * np.pi * positions[16]) * (1 - positions)

fig, (field, section) = plt.subplots(1, 2, figsize=(10, 4))
limit = np.abs(psi).max()
image = field.imshow(psi.T, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu_r', vmin=-limit, vmax=limit)
fig.colorbar(image, ax=field, label='streamfunction')
field.set_xlabel('longitude')
field.set_ylabel('latitude')
field.set_title('Two gyres with western boundary currents')
section.plot(positions, psi[:, 16], label='streamfunction')
section.plot(positions, sverdrup, 'k--', label='Sverdrup balance')
section.set_xlabel('longitude at the centre of the southern gyre')
section.set_ylabel('streamfunction')
section.set_title('The interior follows the Sverdrup balance')
section.legend()
fig.tight_layout()
plt.show()
