"""
# Active nematic gels

An extensile nematic in a viscous fluid drives a flow by its own stress, and this flow bends the axes further, so an
aligned state is unstable. At the default parameters the flow comes in bursts, between which the axes align again.
"""

# %%
# The model is created by name from the catalog on a periodic square of 32 sites. The catalog defaults integrate it with
# RK4 and stream the velocity as the observables `u_x` and `u_y`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import AlignmentAnalysis, DefectAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'active_gels'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = (
    Simulation(
        system,
        dt=1.0,
        solver=RK4Solver(dt=0.05),
        observers=[observer],
        observables=describe(MODEL).observables_of(system),
        random_state=0,
    )
    .check()
    .prepare()
    .run(until=200.0)
)

# %%
# Defects and flow
# ----------------
# The charges of a periodic lattice sum to 0 at every frame, whatever defects a frame holds. The kinetic energy should
# rise from the nearly still start while the instability grows, and then come in bursts.

trajectory = simulation.trajectory_
tensors = trajectory.lattice('gel.Q')
axes = AlignmentAnalysis('principal-axis').check().prepare(tensors).run().axes_
defects = DefectAnalysis(period=np.pi).check().prepare(np.arctan2(axes[..., 1], axes[..., 0])).run()
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
