"""
# Fully biased walk among broad traps

Each row is the landscape of one walker, and the bright part of a row is the stretch of sites the walker has already
crossed. The walkers move only to the right, but a few deep traps hold them for most of the time, so the bright bars
grow in bursts and end at very different lengths.
"""

# %%
# The model is created by name from the catalog with 56 walkers, each in its own line of 80 traps whose trapping times
# are distributed as `tau^(-3/2)`, and simulated for 1495 steps. The observer saves every frame with the default
# blueprint, whose primary view is the lattice of the sites the walkers have visited.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'biased_trap_walk'

system = create(MODEL, n_walkers=56, n_sites=80, alpha=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=5,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=1495)

# %%
# Subdiffusive displacement
# -------------------------
# The time to cross `N` sites is a sum of `N` trapping times of infinite mean, which grows as `N^(1 / alpha)`, so the
# displacement grows as `t^alpha`. Renewal theory gives the mean displacement `(t / tau0)^alpha sin(pi alpha) /
# (pi alpha)` at long times. The mean displacements of 100 walkers for three exponents should approach these lines, and
# the displacements of the walkers above should spread over a broad range, since a single deep trap stops a walker.

fig, (scaling, spread) = plt.subplots(1, 2, figsize=(10, 4))
for alpha in (0.3, 0.5, 0.7):
    model_system = create(MODEL, n_walkers=100, n_sites=400, alpha=alpha, random_state=1)
    run = Simulation(model_system, dt=10, random_state=0, observables=describe(MODEL).observables_of(model_system))
    trajectory = run.run(until=3000).trajectory_
    times, position = trajectory.times[1:], trajectory['position'][1:]
    (line,) = scaling.loglog(times, position, label=f'alpha = {alpha}')
    prediction = times**alpha * np.sin(np.pi * alpha) / (np.pi * alpha)
    scaling.loglog(times, prediction, color=line.get_color(), linestyle='--')
scaling.set_xlabel('time t')
scaling.set_ylabel('mean displacement <x>')
scaling.set_title('Subdiffusion, <x> ~ t^alpha (dashed: renewal theory)')
scaling.legend()
visited = simulation.trajectory_['traps.visited'][-1].reshape(56, 80)
spread.hist(visited.sum(axis=1) - 1, bins=15)
spread.set_xlabel('displacement at t = 1495')
spread.set_ylabel('number of walkers')
spread.set_title('A broad spread of displacements')
fig.tight_layout()
plt.show()
