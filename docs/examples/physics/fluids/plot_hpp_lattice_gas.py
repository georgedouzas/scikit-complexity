"""
# HPP lattice gas

Particles hop along the four directions of a square lattice, and a head-on pair turns by a quarter turn when it meets.
These deterministic, reversible rules conserve particles and momentum, and a dense disc spreads as a sound wave. They
also conserve the momentum of every row and column, so the square gas never becomes an isotropic fluid.
"""

# %%
# The gas fills a periodic 64 x 64 lattice at the density 0.2 per channel, with a denser disc at the centre. The
# default blueprint shows the number of particles at every site.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import HPPLatticeGasSystem

SYSTEM = 'hpp_lattice_gas'

system = HPPLatticeGasSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(system, observers=[observer])

# %%
# Spreading of the dense disc
# ---------------------------
# Averaged over three steps, the excess particles of the disc should run outwards as a sound wave and leave a depleted
# centre behind them, while the total number of particles stays fixed.

density = np.asarray(trajectory.values('gas.density'), dtype=float).reshape(-1, 64, 64)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, step in zip(axes, (0, 10, 20), strict=True):
    ax.imshow(density[step : step + 3].mean(axis=0), cmap='viridis', vmin=0, vmax=2.4)
    ax.set_title(f'steps {step} to {step + 2}')
    ax.set_axis_off()
plt.show()
