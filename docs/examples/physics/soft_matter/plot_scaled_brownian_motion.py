"""
# Scaled Brownian motion

Brownian motion whose diffusion coefficient decays in time as `t^(alpha - 1)`, as in a medium that cools or stiffens.
With `alpha = 0.5` the walkers move fast at first and then slow down, so their paths fan out quickly and then wander
with ever smaller steps, and the spread grows as `t^0.5` instead of `t`. The view shows the paths of eight walkers.
"""

# %%
# The model is created by name from the catalog with eight walkers, `alpha = 0.5`, `K = 1`, and `t0 = 0.01`. The
# Euler-Maruyama stepper takes steps of 0.01 and the observer saves every frame of 0.05 with the default blueprint,
# whose primary view is the time series of the positions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'scaled_brownian_motion'

system = create(MODEL, n_particles=8, alpha=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=0.05, observers=[observer], random_state=0).run(until=10)

# %%
# Anomalous mean squared displacement
# -----------------------------------
# Integrating the diffusion coefficient gives `<x^2> = 2 K ((t + t0)^alpha - t0^alpha)`, close to `2 K t^alpha`
# once `t >> t0`. The mean squared displacements of 2000 walkers should follow these laws for subdiffusion with
# `alpha = 0.5` and superdiffusion with `alpha = 1.5`, on either side of the normal diffusion `2 K t`.

fig, ax = plt.subplots(figsize=(7, 4))
for alpha, color in ((0.5, 'tab:blue'), (1.5, 'tab:orange')):
    run = Simulation(create(MODEL, n_particles=2000, alpha=alpha), solver=solver, dt=0.05, random_state=1).run(10)
    times = run.trajectory_.times[1:]
    squared = np.mean(run.trajectory_['walkers.x'][1:] ** 2, axis=1)
    ax.loglog(times, squared, 'o', markersize=3, color=color, label=f'simulation, alpha = {alpha}')
    exact = 2 * ((times + 0.01) ** alpha - 0.01**alpha)
    ax.loglog(times, exact, color=color, label=f'2 K ((t + t0)^{alpha} - t0^{alpha})')
ax.loglog(times, 2 * times, ':', color='gray', label='normal diffusion, 2 K t')
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Sub- and superdiffusion from a time-dependent diffusivity')
ax.legend()
fig.tight_layout()
plt.show()
