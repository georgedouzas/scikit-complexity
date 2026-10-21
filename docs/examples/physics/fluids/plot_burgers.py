"""
# Inviscid Burgers equation

In the inviscid Burgers equation `u_t + u u_x = 0` every value of `u` travels at its own speed `u`, so the fast crest
of a sine wave catches up with the slow trough ahead of it. The front steepens until it becomes vertical and a shock
forms at the time `1 / amplitude`, after which the profile is a sawtooth whose shock moves at the mean of the values
on either side. The space-time image shows the field, one row per saved step with the earliest at the top, with the
smooth sine turning into a sharp moving jump.
"""

# %%
# The model is created by name from the catalog with 200 finite-volume cells, mean `1.5`, and amplitude `1`. The
# catalog registers the strong-stability-preserving Runge-Kutta stepper for this model, which keeps the shock free of
# new extrema, so it is passed explicitly. The observer saves every frame of 0.01 with the default blueprint, whose
# primary view is the space-time image of the field.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'burgers'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=SSPRK(), dt=0.01, observers=[observer]).run(until=2.99)

# %%
# Shock formation and conservation
# --------------------------------
# Characteristics first cross at `t = 1 / max(-u_0') = 1 / amplitude = 1`, so the steepest slope of the profile should
# grow as `1 / (1 - t)` until then and saturate at the resolution of the grid after it. The finite volumes conserve the
# total of `u`, which should stay at its initial value through the shock.

trajectory = simulation.trajectory_
times = trajectory.times
values = trajectory['line.u']
spacing = 2 * np.pi / values.shape[1]
slope = np.abs(np.diff(values, axis=1, append=values[:, :1])).max(axis=1) / spacing
drift = np.abs(values.sum(axis=1) - values[0].sum()) / values[0].sum()
fig, (steep, total) = plt.subplots(1, 2, figsize=(10, 4))
steep.semilogy(times, slope, label='simulation')
early = times < 0.95
steep.semilogy(times[early], 1 / (1 - times[early]), '--', color='black', label='1 / (1 - t)')
steep.axvline(1.0, color='gray', linestyle=':', label='shock time 1 / amplitude')
steep.set_xlabel('time t')
steep.set_ylabel('steepest slope |u_x|')
steep.set_title('The front steepens into a shock at t = 1')
steep.legend()
total.semilogy(times[1:], drift[1:] + 1e-17)
total.set_xlabel('time t')
total.set_ylabel('relative change of the total of u')
total.set_title('The total is conserved to round-off')
fig.tight_layout()
plt.show()
