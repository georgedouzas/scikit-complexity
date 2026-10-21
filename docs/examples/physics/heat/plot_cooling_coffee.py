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

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cooling_coffee'

system = create(MODEL, T=90.0, room=20.0, k=0.1)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=60)

# %%
# Exponential relaxation
# ----------------------
# The solution is `T(t) = T_room + (T0 - T_room) exp(-k t)`, so the excess temperature `T - T_room` should fall on a
# straight line of slope `-k` on a logarithmic axis, for every starting temperature.

times = np.asarray(simulation.trajectory_.times)
fig, ax = plt.subplots(figsize=(6, 4))
for start in [90.0, 60.0, 40.0]:
    run = Simulation(create(MODEL, T=start, room=20.0, k=0.1), dt=0.5).run(until=60)
    line = ax.semilogy(times, np.ravel(run.trajectory_['coffee.T']) - 20.0, label=f'T0 = {start:.0f}')[0]
    ax.semilogy(times, (start - 20.0) * np.exp(-0.1 * times), '--', color=line.get_color())
ax.set_xlabel('time (minutes)')
ax.set_ylabel('T - T_room')
ax.set_title('Excess temperature decays as exp(-k t), dashed')
ax.legend()
plt.show()
