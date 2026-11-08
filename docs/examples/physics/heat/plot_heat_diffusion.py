"""
# Heat diffusion

A hot square in the middle of an insulated plate spreads into a round, ever wider and lower bump. The total heat stays
the same, and the width of the bump grows like the square root of time, the signature of diffusion.
"""

# %%
# The plate has 64 by 64 sites, zero-flux edges, and the diffusivity `D = 1`. The default blueprint shows the
# temperature as an image, with the hot square rounding off within the first few frames.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.heat import HeatDiffusionSystem

SYSTEM = 'heat_diffusion'

system = HeatDiffusionSystem(shape=(64, 64), D=1.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer])

# %%
# Conservation and spreading
# --------------------------
# With insulated edges the total heat should stay constant. Away from the edges the variance of the temperature
# profile along each axis grows as `var(t) = var(0) + 2 D t`, so the measured variance should follow that line.

fields = trajectory.lattice('plate.u')
times = np.asarray(trajectory.times)
coordinates = np.arange(fields.shape[1])
total = fields.sum(axis=(1, 2))
profile = fields.sum(axis=2) / total[:, None]
mean = profile @ coordinates
variance = profile @ coordinates**2 - mean**2
fig, (heat, spread) = plt.subplots(1, 2, figsize=(10, 4))
heat.plot(times, total / total[0])
heat.set_ylim(0.99, 1.01)
heat.set_xlabel('time')
heat.set_ylabel('total heat / initial total heat')
heat.set_title('The total heat is conserved')
spread.plot(times, variance, label='simulated')
spread.plot(times, variance[0] + 2 * times, '--', color='black', label='var(0) + 2 D t')
spread.set_xlabel('time')
spread.set_ylabel('variance along x')
spread.set_title('Diffusive spreading')
spread.legend()
fig.tight_layout()
plt.show()
