"""
# Heated room

A room is assembled from thermal and electrical components: a voltage source, a switched heating resistor, the air of
the room, and a wall to the outside. A thermostat with hysteresis opens the switch above its band and closes it below,
so the temperature cycles inside the band.
"""

# %%
# The room starts at 10 °C with 5 °C outside and a setting of 20 °C, and runs for four hours. The default blueprint
# shows the temperatures and the power.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'heated_room'

system = create(MODEL)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = (
    Simulation(system, dt=10.0, observers=[observer], observables=observables).check().prepare().run(until=14400.0)
)

# %%
# Temperature inside the band
# ---------------------------
# Once warm, the room stays between 19.5 °C and 20.5 °C while the heater switches on and off.

trajectory = simulation.trajectory_
hours = trajectory.times / 3600
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hours, trajectory.values('room.T'))
ax.axhline(19.5, color='black', linestyle='--')
ax.axhline(20.5, color='black', linestyle='--')
ax.set_xlabel('hours')
ax.set_ylabel('room temperature (°C)')
ax.set_title('A thermostat with hysteresis')
plt.show()
