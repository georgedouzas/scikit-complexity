"""
# Tumour-immune suppression

At a moderate antigenicity the tumour and the T cells oscillate: the tumour grows to about `3 10^6` cells, the T cells
it recruits cut it back, and it regrows once regulatory T cells, IL-10, and TGF-beta have damped the response.
"""

# %%
# The model is created by name from the catalog with an antigenicity of `10^-3` and runs for 200 days from a single
# tumour cell. The observer saves every frame with the default blueprint, whose primary view is the time series of the
# tumour, the immune cells, and the cytokines.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tumour_immune_suppression'

solver = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, solver=solver, observers=[observer]).run(until=200)

# %%
# Antigenicity decides escape
# ---------------------------
# Below an antigenicity of about `10^-4` the tumour should escape, first growing exponentially and then as the square
# root of its size. At `10^-3` it should oscillate, and from `10^-2` on the cytotoxic T cells should hold it to a few
# cells.

fig, (tumour, cells) = plt.subplots(1, 2, figsize=(10, 4))
for antigenicity in (1e-5, 1e-4, 1e-3, 1e-2, 1.0, 100.0):
    trajectory = Simulation(create(MODEL, alpha_a=antigenicity), dt=1.0, solver=solver).run(until=200).trajectory_
    tumour.semilogy(trajectory.times, trajectory['site.A'], label=f'alpha_a = {antigenicity:g}')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells')
tumour.set_title('Weakly antigenic tumours escape')
tumour.legend()
default = simulation.trajectory_
for state, name in (('E', 'cytotoxic T cells'), ('EH', 'helper T cells'), ('ER', 'regulatory T cells')):
    cells.semilogy(default.times, default[f'site.{state}'].clip(1e-6), label=name)
cells.set_xlabel('time (days)')
cells.set_ylabel('cells')
cells.set_title('The response at alpha_a = 0.001')
cells.legend()
fig.tight_layout()
plt.show()
