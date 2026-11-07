"""
# Wilson-Cowan

An excitatory and an inhibitory population of neurons, each driving the other, oscillate under a moderate input: the
excitatory activity rises, recruits the inhibition, which shuts it down, and the cycle repeats as a limit cycle of
population activity.
"""

# %%
# The system is built from its class with Wilson and Cowan's couplings and the input `P = 1.25`. The observer saves
# every second frame with the default blueprint, whose primary view is the time series of both activities.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import WilsonCowanSystem

SYSTEM = 'wilson_cowan'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(WilsonCowanSystem(P=1.25), observers=[observer], until=50)

# %%
# Limit cycle against rest
# ------------------------
# With the input the trajectory should settle on a closed orbit in the `(E, I)` plane whatever its start, while
# without input the activity should decay to rest.

oscillating = trajectory
second = simulate(WilsonCowanSystem(E=0.4, I=0.3, P=1.25))
resting = simulate(WilsonCowanSystem(E=0.4, I=0.3, P=0.0))
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(oscillating.values('cortex.E'), oscillating.values('cortex.I'), label='P = 1.25, from (0.1, 0.05)')
ax.plot(second.values('cortex.E'), second.values('cortex.I'), '--', label='P = 1.25, from (0.4, 0.3)')
ax.plot(resting.values('cortex.E'), resting.values('cortex.I'), ':', color='black', label='P = 0, decays to rest')
ax.set_xlabel('excitatory activity E')
ax.set_ylabel('inhibitory activity I')
ax.set_title('An input turns rest into a limit cycle')
ax.legend()
plt.show()
