"""
# Brusselator

In a well-mixed reactor with `b = 3 > 1 + a^2`, the Brusselator leaves its unstable fixed point and settles on a limit
cycle: the concentrations `u` and `v` oscillate out of phase with a fixed amplitude, whatever the initial state.
"""

# %%
# The model is created by name from the catalog with `a = 1` and `b = 3`, starting at `u = v = 1`, and runs for about
# seven periods. The default blueprint shows both concentrations in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'brusselator_ode'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# One limit cycle from every initial state
# ----------------------------------------
# The fixed point `(a, b / a) = (1, 3)` is unstable for `b > 1 + a^2`. Runs started inside and outside the cycle
# should all wind onto the same closed orbit, the one the run above reaches.

fig, ax = plt.subplots(figsize=(6, 5))
for u0, v0 in [(1.0, 3.2), (0.5, 0.5), (4.0, 1.0), (3.0, 5.0)]:
    run = Simulation(create(MODEL, u=u0, v=v0), dt=0.05).run(until=100).trajectory_
    ax.plot(run['reactor.u'], run['reactor.v'], linewidth=0.8, alpha=0.7)
    ax.plot(u0, v0, 'o', color='gray', markersize=4)
trajectory = simulation.trajectory_
late = trajectory.times > 30
ax.plot(trajectory['reactor.u'][late], trajectory['reactor.v'][late], color='black', linewidth=2, label='limit cycle')
ax.plot(1.0, 3.0, 'x', color='red', markersize=10, label='unstable fixed point')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_title('All initial states converge to the limit cycle')
ax.legend()
plt.show()
