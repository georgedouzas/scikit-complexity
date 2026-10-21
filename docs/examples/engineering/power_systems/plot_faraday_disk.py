"""
# Faraday disk charger

A motor spins up a Faraday disk until the friction balances the torque of the motor. The disk then turns at a steady
speed, but its voltage stays below that of the battery, so after a first surge the current settles at a small negative
value: the battery discharges slowly through the disk.
"""

# %%
# The model is created by name from the catalog with the field `B = 1.5` and the motor torque `M = 4.8`, starting at
# rest, and runs for 50 time units. The default blueprint shows the current, the polarization voltage of the battery,
# and the angular speed in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'faraday_disk'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Steady states against the torque
# --------------------------------
# From rest, the speed and the current should settle on a stable steady state, where the trajectory in the plane of
# speed and current ends. Over a range of torques the steady speed should grow almost linearly with the torque, as the
# viscous friction `beta r2 omega` takes up most of it. Above a torque near 9.5 the self-excitation of the coil makes
# the current oscillate instead, which the sweep stops short of.

torques = np.linspace(3.0, 9.0, 13)
speeds = [
    Simulation(create(MODEL, M=torque), dt=0.1).run(until=50).trajectory_['charger.omega'][-1] for torque in torques
]
trajectory = simulation.trajectory_
fig, (settling, speed) = plt.subplots(1, 2, figsize=(10, 4))
settling.plot(trajectory['charger.omega'], trajectory['charger.I'])
settling.plot(trajectory['charger.omega'][-1], trajectory['charger.I'][-1], 'o', color='black', label='steady state')
settling.set_xlabel('angular speed omega')
settling.set_ylabel('current I')
settling.set_title('Spin-up to a steady state')
settling.legend()
speed.plot(torques, speeds, 'o-', markersize=3)
speed.set_xlabel('motor torque M')
speed.set_ylabel('steady angular speed')
speed.set_title('The disk turns faster with more torque')
fig.tight_layout()
plt.show()
