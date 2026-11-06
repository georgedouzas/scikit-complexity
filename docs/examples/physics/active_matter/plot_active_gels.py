"""
# Active nematic gels

An extensile nematic in a viscous fluid drives a flow by its own stress, and this flow bends the axes further, so an
aligned state is unstable. Pairs of `+1/2` and `-1/2` defects appear and move in a chaotic flow, active turbulence.
"""

# %%
# The system is created by name from the catalog on a periodic square of 128 sites and side 10, at the parameters of the
# chaotic state of Giomi et al. 2012. The catalog defaults integrate it with RK4 and stream the velocity as the
# observables `u_x` and `u_y`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import AlignmentAnalysis, DefectAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system, describe

SYSTEM = 'active_gels'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = (
    Simulation(
        system,
        dt=0.025,
        solver=RK4Solver(dt=5e-4),
        observers=[observer],
        observables=describe(SYSTEM).observables_of(system),
        random_state=0,
    )
    .prepare()
    .run(until=0.5)
)

# %%
# Defects and flow
# ----------------
# Every defect of a periodic lattice has a partner of the opposite charge, so the total charge should stay 0 at every
# frame. The kinetic energy should rise from the nearly still start while the instability grows, and then settle.

trajectory = simulation.trajectory_
tensors = trajectory.lattice('gel.Q')
axes = AlignmentAnalysis('principal-axis').prepare(tensors).run().axes_
defects = DefectAnalysis(period=np.pi).prepare(np.arctan2(axes[..., 1], axes[..., 0])).run()
energy = 0.5 * (np.asarray(trajectory.values('u_x')) ** 2 + np.asarray(trajectory.values('u_y')) ** 2).mean(axis=1)

fig, (count, kinetic) = plt.subplots(1, 2, figsize=(10, 4))
count.plot(trajectory.times, [len(item) for item in defects.charges_])
count.set_xlabel('time')
count.set_ylabel('number of defects')
kinetic.plot(trajectory.times, energy)
kinetic.set_xlabel('time')
kinetic.set_ylabel('mean kinetic energy')
fig.tight_layout()
assert max(abs(item.sum()) for item in defects.charges_) < 1e-6
