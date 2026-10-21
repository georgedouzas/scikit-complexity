"""
# Wilson-Cowan

An excitatory and an inhibitory population of neurons, each driving the other, oscillate under a moderate input: the
excitatory activity rises, recruits the inhibition, which shuts it down, and the cycle repeats as a limit cycle of
population activity.
"""

# %%
# The model is created by name from the catalog with Wilson and Cowan's couplings and the input `P = 1.25`. The
# observer saves every second frame with the default blueprint, whose primary view is the time series of both
# activities.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'wilson_cowan'

system = create(MODEL, P=1.25)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=50)

# %%
# Limit cycle against rest
# ------------------------
# With the input the trajectory should settle on a closed orbit in the `(E, I)` plane whatever its start, while
# without input the activity should decay to rest.

oscillating = simulation.trajectory_
second = Simulation(create(MODEL, E=0.4, I=0.3, P=1.25), dt=0.2).run(until=100).trajectory_
resting = Simulation(create(MODEL, E=0.4, I=0.3, P=0.0), dt=0.2).run(until=100).trajectory_
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(oscillating['cortex.E'], oscillating['cortex.I'], label='P = 1.25, from (0.1, 0.05)')
ax.plot(second['cortex.E'], second['cortex.I'], '--', label='P = 1.25, from (0.4, 0.3)')
ax.plot(resting['cortex.E'], resting['cortex.I'], ':', color='black', label='P = 0, decays to rest')
ax.set_xlabel('excitatory activity E')
ax.set_ylabel('inhibitory activity I')
ax.set_title('An input turns rest into a limit cycle')
ax.legend()
plt.show()
