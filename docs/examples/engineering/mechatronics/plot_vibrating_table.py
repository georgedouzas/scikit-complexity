"""
# Electromagnetic vibrating table

A shaker table hangs on a spring and a damper and is pulled by an electromagnet, whose coil a sinusoidal source feeds
through a resistor and a capacitor. After the switch closes, a transient of a few seconds dies out and the table
vibrates steadily at the frequency of the supply. The linearized model of Trzaska gives nearly the same response.
"""

# %%
# The system is built from its class with Trzaska's parameters, `u = 0.141 sin(1.5915 t)`, `R = 1.5`,
# `L = 0.25`, `C = 0.1`, `K = 500`, `m = 10`, `B = 2`, `k1 = 0.5`, `k2 = 17`, and `alpha = 2.5`, and starts at rest.
# The default blueprint shows the current, the capacitor voltage, and the motion of the table.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.mechatronics import VibratingTableSystem

SYSTEM = 'vibrating_table'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
trajectory = simulate(VibratingTableSystem(), observers=[observer])

# %%
# Nonlinear and linearized models
# -------------------------------
# The displacement should settle on a sinusoid after about 8 s, and the phase trajectory of the displacement against
# the current should close on an ellipse, nearly the same for the nonlinear and the linearized electromagnet.

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
for linear in (False, True):
    trajectory = simulate(VibratingTableSystem(linear=linear))
    times = trajectory.times
    x, current = np.ravel(trajectory.values('shaker.x')), np.ravel(trajectory.values('shaker.i2'))
    label = 'linearized' if linear else 'nonlinear'
    left.plot(times, x, label=label)
    late = times > 20
    right.plot(current[late], x[late], label=label)
left.set_xlabel('time [s]')
left.set_ylabel('displacement x [m]')
left.legend()
right.set_xlabel('current i2 [A]')
right.set_ylabel('displacement x [m]')
right.set_title('Steady phase trajectory')
plt.show()
