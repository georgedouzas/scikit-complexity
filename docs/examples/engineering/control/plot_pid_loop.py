"""
# PID control loop

The controller drives the output of a work process to its set point. When a disturbance steps in, the output dips,
and the integral action pulls it back to the set point instead of leaving a lasting error.
"""

# %%
# The model is created by name from the catalog with `Kp = 2`, `Ki = 1`, `Kd = 0.1`, and the set point 1. The
# disturbance -0.5 steps in at time 10. The default blueprint shows the output and the integral of the error in one time
# series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pid_loop'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=40)

# %%
# Why the integral action matters
# -------------------------------
# Under proportional control alone, `Ki = 0`, the output should settle below the set point, at
# `gain Kp r / (1 + gain Kp) = 0.67` before the disturbance and lower after it. With the integral action the output
# should return to the set point 1 after every step.

fig, ax = plt.subplots(figsize=(7, 4.5))
trajectory = simulation.trajectory_
ax.plot(trajectory.times, trajectory['process.y'], label='PID, Ki = 1')
proportional = Simulation(create(MODEL, Ki=0.0, Kd=0.0), dt=0.1).run(until=40).trajectory_
ax.plot(proportional.times, proportional['process.y'], label='P only, Ki = 0')
ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, label='set point')
ax.axhline(2 / 3, color='gray', linestyle=':', linewidth=0.8, label='P-only level 2/3')
ax.axvline(10.0, color='C3', linewidth=0.8, label='disturbance')
ax.set_xlabel('time')
ax.set_ylabel('output y')
ax.set_title('The integral action removes the steady error')
ax.legend()
plt.show()
