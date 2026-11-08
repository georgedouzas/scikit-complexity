"""
# Active nematic gels

An extensile nematic in a viscous fluid drives a flow by its own stress, and this flow bends the axes further, so an
aligned state is unstable. Pairs of `+1/2` and `-1/2` defects appear and move in a chaotic flow, active turbulence.
"""

# %%
# The system is built from its class on a periodic square of 128 sites and side 10, at the parameters of the
# chaotic state of Giomi et al. 2012. The catalog defaults integrate it with RK4 and stream the velocity as the
# observables `u_x` and `u_y`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import alignment, defects
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import ActiveGelsSystem

SYSTEM = 'active_gels'

system = ActiveGelsSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=0.5, dt=0.025, solver=RK4Solver(dt=5e-4), observers=[observer], random_state=0)

# %%
# Defects and flow
# ----------------
# Every defect of a periodic lattice has a partner of the opposite charge, so the total charge should stay 0 at every
# frame. The kinetic energy should rise from the nearly still start while the instability grows, and then settle.

tensors = trajectory.lattice('gel.Q')
axes = alignment(tensors, kind='principal-axis').axes
found = defects(np.arctan2(axes[..., 1], axes[..., 0]), period=np.pi)
energy = 0.5 * (np.asarray(trajectory.values('u_x')) ** 2 + np.asarray(trajectory.values('u_y')) ** 2).mean(axis=1)

fig, (count, kinetic) = plt.subplots(1, 2, figsize=(10, 4))
count.plot(trajectory.times, [len(item) for item in found.charges])
count.set_xlabel('time')
count.set_ylabel('number of defects')
kinetic.plot(trajectory.times, energy)
kinetic.set_xlabel('time')
kinetic.set_ylabel('mean kinetic energy')
fig.tight_layout()
assert max(abs(item.sum()) for item in found.charges) < 1e-6
