"""
# Dry active nematics

A dense layer of elongated, self-driven objects on a substrate aligns its axes without any fluid. An active current
along the divergence of the alignment tensor moves the density where the axes bend, which no free energy gives. The
image shows the density at the end of the run.
"""

# %%
# The model is created by name from the catalog on a periodic square of 64 sites, above the critical density, where the
# isotropic state loses stability. The catalog defaults integrate it with RK4 on the Numba backend.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'dry_active_nematics'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = (
    Simulation(system, dt=5.0, solver=RK4Solver(dt=0.02), backend=NumbaBackend(), observers=[observer], random_state=0)
    .check()
    .prepare()
    .run(until=100.0)
)

# %%
# Order and conservation
# ----------------------
# The scalar order should rise from the small initial noise to a nonzero value, and the total density should stay at its
# initial value to rounding, since the current is a divergence.

trajectory = simulation.trajectory_
alignment = AlignmentAnalysis('principal-axis').check().prepare(trajectory.lattice('nematic.Q')).run()
density = trajectory.lattice('nematic.rho')

fig, (order, total) = plt.subplots(1, 2, figsize=(10, 4))
order.plot(trajectory.times, alignment.order_.mean(axis=(1, 2)))
order.set_xlabel('time')
order.set_ylabel('mean scalar order')
total.plot(trajectory.times, density.sum(axis=(1, 2)) - density[0].sum())
total.set_xlabel('time')
total.set_ylabel('change of the total density')
fig.tight_layout()
assert np.abs(density.sum(axis=(1, 2)) - density[0].sum()).max() < 1e-9 * density[0].sum()
