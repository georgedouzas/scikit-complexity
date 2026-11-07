"""
# Duffing

A forced, damped particle in a double-well potential never settles: it hops irregularly between the two wells at
`x = -1` and `x = 1`, and the stroboscopic picture of its motion is a folded strange attractor.
"""

# %%
# The parameters are the classic chaotic ones, `delta = 0.3`, `alpha = -1`, `beta = 1`, `gamma = 0.5`,
# `omega = 1.2`. The default blueprint shows the position and the velocity as time series over the whole run of about
# forty forcing periods.

import math

import matplotlib.pyplot as plt

from skcomplex.core.analysis import section
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.nonlinear_oscillators import DuffingSystem

SYSTEM = 'duffing'

system = DuffingSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=16)
trajectory = simulate(system, observers=[observer])

# %%
# Stroboscopic section
# --------------------
# Sampling the state once per forcing period `2π / ω` removes the drive from the picture. A periodic response would
# leave a few isolated points, while the chaotic Duffing oscillator should fill a folded, layered set, Ueda's strange
# attractor, with points in both wells.

points = section(
    DuffingSystem().prepare().system_,
    kind='stroboscopic',
    period=2 * math.pi / 1.2,
    n_points=3000,
    transient=50,
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(points.values['duffing.x'], points.values['duffing.x_d1'], ',', color='black', alpha=0.6)
ax.set_xlabel('x')
ax.set_ylabel("x'")
ax.set_title('Stroboscopic section: a strange attractor')
plt.show()
