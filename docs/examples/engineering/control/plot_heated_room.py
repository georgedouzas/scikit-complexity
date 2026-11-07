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

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.control import HeatedRoomSystem

SYSTEM = 'heated_room'

system = HeatedRoomSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
trajectory = simulate(system, observers=[observer])

# %%
# Temperature inside the band
# ---------------------------
# Once warm, the room stays between 19.5 °C and 20.5 °C while the heater switches on and off.

hours = trajectory.times / 3600
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hours, trajectory.values('room.T'))
ax.axhline(19.5, color='black', linestyle='--')
ax.axhline(20.5, color='black', linestyle='--')
ax.set_xlabel('hours')
ax.set_ylabel('room temperature (°C)')
ax.set_title('A thermostat with hysteresis')
plt.show()
