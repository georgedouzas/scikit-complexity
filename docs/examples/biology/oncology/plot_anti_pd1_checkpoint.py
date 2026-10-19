"""
# Anti-PD-1 checkpoint inhibition

At the book's low anti-PD-1 infusion, PD-1-PD-L1 binding keeps the T cells low, and the tumour falls at first and then
grows back toward its capacity.
"""

# %%
# The model is created by name from the catalog with the book's continuous infusion of `10^-10` g/cm³ per day and runs
# for 150 days. The observer saves every frame with the default blueprint, whose primary view is the time series of the
# T cells, the tumour, and the drug.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'anti_pd1_checkpoint'

solver = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, solver=solver, observers=[observer]).run(until=150)

# %%
# Dose response
# -------------
# Without the drug the T cells decay to a low level and the tumour grows toward its capacity. A tenfold larger infusion
# should leave more T cells and a smaller tumour. At twenty times the book's infusion the drug blocks all PD-1, and the
# T cells should grow without bound and clear the tumour.

runs = {dose: Simulation(create(MODEL, u_ap=dose), dt=1.0, solver=solver).run(until=150) for dose in (0.0, 1e-9, 2e-9)}
runs[1e-10] = simulation
fig, (tumour, t_cells) = plt.subplots(1, 2, figsize=(10, 4))
for dose, run in sorted(runs.items()):
    trajectory = run.trajectory_
    tumour.plot(trajectory.times, trajectory['site.A'], label=f'u = {dose:g}')
    t_cells.semilogy(trajectory.times, trajectory['site.E'], label=f'u = {dose:g}')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour (g/cm³)')
tumour.set_title('More drug, smaller tumour')
tumour.legend()
t_cells.set_xlabel('time (days)')
t_cells.set_ylabel('activated T cells (g/cm³)')
t_cells.set_title('More drug, more T cells')
t_cells.legend()
fig.tight_layout()
plt.show()
