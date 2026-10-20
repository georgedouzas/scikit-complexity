"""
# Bouchaud trap model

Each point is a walker in a trap, placed at random in the plane and colored by its depth. Below the glass temperature
the walkers find ever deeper traps, so the points darken less and less often and the escapes slow down as the system
ages.
"""

# %%
# The model is created by name from the catalog with 500 walkers at the temperature `T = 0.5`, with trap depths of
# density `exp(-E)` and escape times of mean `exp(E / T)`, and simulated to time 10000. Every walker renews on its own
# clock, with no time grid. The observer saves every frame with the default blueprint, whose primary view shows the
# walkers in their traps.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'trap_model'

system = create(MODEL, n_walkers=500, T=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=50, observers=[observer], random_state=0).run(until=10000)

# %%
# Aging below the glass temperature
# ---------------------------------
# The escape times have the tail `tau^(-1 - T)`, whose mean diverges for `T < 1`. The number of escapes per walker
# should then grow as `t^T`, along the dashed line of slope 0.5, instead of linearly, and the mean depth of the occupied
# traps should keep growing as `T ln t` instead of settling at the mean depth 1 of the landscape.

trajectory = simulation.trajectory_
times = trajectory.times[1:]
escapes = trajectory['escapes.renewals'][1:] / 500
depth = np.mean(trajectory['walkers.E'][1:], axis=1)
late = times > 500
slope = np.polyfit(np.log(times[late]), np.log(escapes[late]), 1)[0]
fig, (counts, aging) = plt.subplots(1, 2, figsize=(10, 4))
counts.loglog(times, escapes, label=f'simulation, slope {slope:.2f} after t = 500')
counts.loglog(times, escapes[-1] * (times / times[-1]) ** 0.5, '--', color='black', label='slope T = 0.5')
counts.set_xlabel('time t')
counts.set_ylabel('escapes per walker')
counts.set_title('Escapes grow as t^T')
counts.legend()
aging.semilogx(times, depth, label='mean depth of the occupied traps')
aging.axhline(1.0, color='gray', linestyle=':', label='mean depth of the landscape')
aging.set_xlabel('time t')
aging.set_ylabel('depth E')
aging.set_title('The walkers age in ever deeper traps')
aging.legend()
fig.tight_layout()
plt.show()
