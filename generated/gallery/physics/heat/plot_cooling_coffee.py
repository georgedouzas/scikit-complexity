"""
# Cooling coffee

A hot cup of coffee cools fast at first and then ever more slowly: its excess temperature over the room decays
exponentially, Newton's law of cooling, and it approaches the room temperature without ever crossing it.
"""

# %%
# The coffee starts at 90 degrees in a room at 20 degrees, with the cooling rate `k = 0.1` per minute. The default
# blueprint shows the temperature over one hour.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.heat import CoolingCoffeeSystem

SYSTEM = 'cooling_coffee'

system = CoolingCoffeeSystem(T=90.0, room=20.0, k=0.1)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer])

# %%
# Exponential relaxation
# ----------------------
# The solution is `T(t) = T_room + (T0 - T_room) exp(-k t)`, so the excess temperature `T - T_room` should fall on a
# straight line of slope `-k` on a logarithmic axis, for every starting temperature.

times = np.asarray(trajectory.times)
fig, ax = plt.subplots(figsize=(6, 4))
for start in [90.0, 60.0, 40.0]:
    run = simulate(CoolingCoffeeSystem(T=start, room=20.0, k=0.1))
    line = ax.semilogy(times, np.ravel(run.get_values('coffee.T')) - 20.0, label=f'T0 = {start:.0f}')[0]
    ax.semilogy(times, (start - 20.0) * np.exp(-0.1 * times), '--', color=line.get_color())
ax.set_xlabel('time (minutes)')
ax.set_ylabel('T - T_room')
ax.set_title('Excess temperature decays as exp(-k t), dashed')
ax.legend()
plt.show()
