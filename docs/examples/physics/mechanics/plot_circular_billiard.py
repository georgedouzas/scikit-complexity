"""
# Circular billiard

A particle bouncing inside a circle hits the wall at the same angle every time, so its path is a regular star of chords
that never enters an inner circle, the caustic. The circular billiard is integrable: no chaos, only rotation.
"""

# %%
# The default blueprint draws the wall of the table and the whole path of the particle.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'circular_billiard'


system = create(MODEL, x=0.3, y=0.0, vx=0.4, vy=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=20)

# %%
# The caustic
# -----------
# Angular momentum `L = x v_y - y v_x` is conserved between bounces and by every reflection, so each chord passes at
# the distance `|L| / |v|` from the centre. The distance of the particle from the centre should never drop below
# that radius, and its histogram should pile up at the caustic, where the chords are tangent.

x, y = np.ravel(simulation.trajectory_['particle.x']), np.ravel(simulation.trajectory_['particle.y'])
vx, vy = np.ravel(simulation.trajectory_['particle.vx']), np.ravel(simulation.trajectory_['particle.vy'])
momentum = x * vy - y * vx
caustic = abs(momentum[0]) / np.hypot(0.4, 1.0)
fig, (conserved, radii) = plt.subplots(1, 2, figsize=(10, 4))
conserved.plot(simulation.trajectory_.times, momentum)
conserved.set_ylim(0, 0.5)
conserved.set_xlabel('time')
conserved.set_ylabel('angular momentum L')
conserved.set_title('L is conserved by every bounce')
radii.hist(np.hypot(x, y), bins=40, density=True, alpha=0.6)
radii.axvline(caustic, color='black', linestyle='--', label=f'caustic |L| / |v| = {caustic:.3f}')
radii.set_xlabel('distance from the centre')
radii.set_ylabel('probability density')
radii.set_title('The path never enters the caustic')
radii.legend()
fig.tight_layout()
plt.show()
