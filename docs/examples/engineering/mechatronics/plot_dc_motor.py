"""
# DC motor

A voltage source drives a resistor, an inductor, and the armature of a motor in series. The current exerts a torque on
a rotor with viscous friction, and the spinning armature opposes the source with a voltage of its own, so the speed
settles at `k V / (R b + k^2)`.
"""

# %%
# The default motor runs from rest for ten seconds. The default blueprint shows the speed and the current.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'dc_motor'

V, R, k, b = 12.0, 1.0, 0.5, 0.2
system = create(MODEL, V=V, R=R, k=k, b=b)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.05, observers=[observer], observables=observables).prepare().run(until=10.0)

# %%
# Speed and current
# -----------------
# The current peaks while the motor starts against little opposing voltage, then falls as the speed rises to the closed
# form.

trajectory = simulation.trajectory_
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(trajectory.times, trajectory.values('speed'), label='speed')
ax.plot(trajectory.times, trajectory.values('current'), label='current')
ax.axhline(k * V / (R * b + k**2), color='black', linestyle='--', label='closed form of the speed')
ax.set_xlabel('time')
ax.set_title('A motor starting from rest')
ax.legend()
plt.show()
