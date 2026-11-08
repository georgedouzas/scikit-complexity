"""
# Dry active nematics

A dense layer of elongated, self-driven objects on a substrate aligns its axes without any fluid. An active current
along the divergence of the alignment tensor moves the density where the axes bend, which no free energy gives. The
image shows the density at the end of the run.
"""

# %%
# The system is built from its class on a periodic square of 64 sites, above the critical density, where the
# isotropic state loses stability. The catalog defaults integrate it with RK4 on the Numba backend.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import alignment
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import DryActiveNematicsSystem

SYSTEM = 'dry_active_nematics'

system = DryActiveNematicsSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, backend=NumbaBackend(), observers=[observer], random_state=0)

# %%
# Order and conservation
# ----------------------
# The scalar order should rise from the small initial noise to a nonzero value, and the total density should stay at its
# initial value to rounding, since the current is a divergence.

tensors = alignment(trajectory.lattice('nematic.Q'), kind='principal-axis')
density = trajectory.lattice('nematic.rho')

fig, (order, total) = plt.subplots(1, 2, figsize=(10, 4))
order.plot(trajectory.times, tensors.order.mean(axis=(1, 2)))
order.set_xlabel('time')
order.set_ylabel('mean scalar order')
total.plot(trajectory.times, density.sum(axis=(1, 2)) - density[0].sum())
total.set_xlabel('time')
total.set_ylabel('change of the total density')
fig.tight_layout()
assert np.abs(density.sum(axis=(1, 2)) - density[0].sum()).max() < 1e-9 * density[0].sum()
