"""
# Mixed-mode Chua circuit

On three time scales, the modified Chua circuit alternates between a few small oscillations and a large relaxation
spike, over and over: a mixed-mode oscillation.
"""

# %%
# The model is created by name from the catalog with `epsilon = 0.01` and `c = 0.00053`, integrated with the Radau
# method for the stiff fast variable, and runs for 200 time units. The default blueprint shows `x`, `y`, and `z` in one
# time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'mixed_mode_chua'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, solver=ScipyODE(method='Radau'), observers=[observer]).run(until=200)

# %%
# Small, large, and mixed
# -----------------------
# The slow drift `c` selects the regime: for `c = 0` the fast variable `x` should make only small oscillations, for
# `c = 0.00105` only large ones, and for `c = 0.00053` small and large oscillations should alternate.

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for ax, c in zip(axes, [0.0, 0.00053, 0.00105], strict=True):
    run = Simulation(create(MODEL, c=c), dt=0.25, solver=ScipyODE(method='Radau')).run(until=200).trajectory_
    ax.plot(run.times, run['circuit.x'], linewidth=0.8)
    ax.set_ylabel('x')
    ax.set_title(f'c = {c}', fontsize=10)
axes[-1].set_xlabel('time')
fig.suptitle('Small, mixed-mode, and large oscillations')
fig.tight_layout()
plt.show()
