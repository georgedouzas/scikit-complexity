"""
# Forced Toda oscillator

A damped oscillator with an exponential spring, stiff when compressed and soft when stretched, is driven periodically.
Its response leans far to the soft side, and under a stronger drive it goes through period doublings and chaos.
"""

# %%
# The oscillator has the damping `d = 0.1` and the driving amplitude `a = 0.88`. The default blueprint shows the
# displacement and the velocity over the run.

import math

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import section
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'forced_toda'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=16)
simulation = Simulation(system, dt=0.05, observers=[observer]).prepare().run(until=300)

# %%
# Orbit diagram
# -------------
# With the stronger driving amplitude `a = 2`, the displacement is sampled once per driving period `T = 2π / ω` after
# a transient, each run starting from the final state of the previous one. A single point is a response of period `T`,
# two points a period-doubled response, and a smear of points a chaotic one.

fig, ax = plt.subplots(figsize=(6, 4))
x, v = 0.0, 0.0
for period in np.linspace(2.0, 6.0, 41):
    omega = 2 * math.pi / period
    system = create_system(SYSTEM, x=x, v=v, a=2.0, omega=omega)
    points = section(system, kind='stroboscopic', period=period, n_points=30, transient=60)
    values = np.asarray(points.values['toda.x'])
    x, v = float(values[-1]), float(np.asarray(points.values['toda.x_d1'])[-1])
    ax.plot(np.full(values.size, period), values, ',', color='black')
ax.set_xlabel('driving period T')
ax.set_ylabel('x at multiples of T')
ax.set_title('Orbit diagram of the forced Toda oscillator, a = 2')
plt.show()
