"""
# Brusselator

In a well-mixed reactor with `b = 3 > 1 + a^2`, the Brusselator leaves its unstable fixed point and settles on a limit
cycle: the concentrations `u` and `v` oscillate out of phase with a fixed amplitude, whatever the initial state.
"""

# %%
# The system is built from its class with `a = 1` and `b = 3`, starting at `u = v = 1`, and runs for about
# seven periods. The default blueprint shows both concentrations in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import BrusselatorODESystem

SYSTEM = 'brusselator_ode'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(BrusselatorODESystem(), observers=[observer])

# %%
# One limit cycle from every initial state
# ----------------------------------------
# The fixed point `(a, b / a) = (1, 3)` is unstable for `b > 1 + a^2`. Runs started inside and outside the cycle
# should all wind onto the same closed orbit, the one the run above reaches.

fig, ax = plt.subplots(figsize=(6, 5))
for u0, v0 in [(1.0, 3.2), (0.5, 0.5), (4.0, 1.0), (3.0, 5.0)]:
    run = simulate(BrusselatorODESystem(u=u0, v=v0), until=100, dt=0.05)
    ax.plot(run.values('reactor.u'), run.values('reactor.v'), linewidth=0.8, alpha=0.7)
    ax.plot(u0, v0, 'o', color='gray', markersize=4)
late = trajectory.times > 30
ax.plot(
    trajectory.values('reactor.u')[late],
    trajectory.values('reactor.v')[late],
    color='black',
    linewidth=2,
    label='limit cycle',
)
ax.plot(1.0, 3.0, 'x', color='red', markersize=10, label='unstable fixed point')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_title('All initial states converge to the limit cycle')
ax.legend()
plt.show()
