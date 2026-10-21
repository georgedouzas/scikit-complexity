"""
# Thermostat

The furnace warms a cold room towards the setting of the thermostat, but heat leaks outside all the time, so the room
settles a little below the setting, at the temperature where the furnace makes up for the leak.
"""

# %%
# The model is created by name from the catalog with the room and the outside at 10 °C, the setting 18 °C, the leak
# 0.1 per hour, and a daily swing of 5 °C outside, and runs for two days. The default blueprint shows the room
# temperature in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'thermostat'

system = create(MODEL, swing=5.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.25, observers=[observer]).run(until=48)

# %%
# The steady room temperature
# ---------------------------
# For a constant outside temperature, the room should settle at `(setting + leak outside) / (1 + leak)`, below the
# setting by a gap that grows with the leak. Runs with several leaks, without the daily swing, should end on that line.

leaks = np.linspace(0.02, 0.5, 13)
steady = [Simulation(create(MODEL, leak=leak), dt=0.25).run(until=72).trajectory_['room.T'][-1] for leak in leaks]
grid = np.linspace(0.0, 0.5, 100)
fig, (day, theory) = plt.subplots(1, 2, figsize=(10, 4))
day.plot(simulation.trajectory_.times, simulation.trajectory_['room.T'], label='room')
day.axhline(18.0, color='black', linestyle='--', linewidth=0.8, label='setting')
day.set_xlabel('time (hours)')
day.set_ylabel('temperature (°C)')
day.set_title('Two days with a swinging outside temperature')
day.legend()
theory.plot(grid, (18.0 + grid * 10.0) / (1 + grid), color='black', label='(setting + leak outside) / (1 + leak)')
theory.plot(leaks, steady, 'o', label='simulation')
theory.set_xlabel('leak per hour')
theory.set_ylabel('steady room temperature (°C)')
theory.set_title('The leak sets the gap below the setting')
theory.legend()
fig.tight_layout()
plt.show()
