"""
# Wheel-rail contact

Each bogie that passes presses the rail down twice, once per wheel. The rail dips under each wheel, rings briefly on
its pad, and comes back to rest before the next bogie arrives, so the deflection repeats with every bogie.
"""

# %%
# The model is created by name from the catalog with wheel loads of 1000 N, one bogie every 10 time units with its
# wheels 3 apart, and a linear ballast, and runs for 60 time units. The default blueprint shows the deflections of the
# rail and the sleeper with their rates.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'wheel_rail_contact'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=60)

# %%
# A linear track
# --------------
# With the nonlinear ballast switched off, `k2 = 0`, the deflection should be proportional to the load: the largest
# rail deflection under every load should lie on a line through the origin, and the response to one bogie should have
# two dips, one per wheel.

trajectory = simulation.trajectory_
loads = np.linspace(250.0, 2000.0, 8)
peaks = [np.abs(Simulation(create(MODEL, F=load), dt=0.05).run(until=10).trajectory_['rail.z']).max() for load in loads]
fig, (bogie, linear) = plt.subplots(1, 2, figsize=(10, 4))
window = (trajectory.times >= 10) & (trajectory.times < 20)
bogie.plot(trajectory.times[window], trajectory['rail.z'][window], label='rail')
bogie.plot(trajectory.times[window], trajectory['rail.zs'][window], label='sleeper')
bogie.set_xlabel('time')
bogie.set_ylabel('deflection')
bogie.set_title('One bogie: one dip per wheel')
bogie.legend()
linear.plot(loads, peaks, 'o', label='simulation')
linear.plot(loads, loads * peaks[-1] / loads[-1], color='black', linewidth=0.8, label='proportional to the load')
linear.set_xlabel('wheel load F (N)')
linear.set_ylabel('largest rail deflection')
linear.set_title('Deflection grows linearly with the load')
linear.legend()
fig.tight_layout()
plt.show()
