"""
# Magnetic levitation under alternating impulses

A small conductive ball moves in a vacuum under a field that pushes and pulls it with short impulses of alternating
sign. Between the impulses its velocity relaxes, `v' + b v = 0`, and the impulses at the times `4 k - 1` and `4 k + 1`
change it by `m` and `-m`. The velocity is a periodic, discontinuous wave that Trzaska writes in closed form with a
triangular and a rectangular wave.
"""

# %%
# The system is created by name from the catalog with `b = 1` and `m = 0.2`, and starts on its periodic solution. The
# default blueprint shows the velocity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'magnetic_levitation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# The closed-form periodic solution
# ---------------------------------
# With `τ(t) = (2 / π) arcsin(sin(π t / 2))` and `e(t) = sign(cos(π t / 2))`, the periodic velocity is
# `v = m / (2 cosh b) exp(-b τ e) e`. The simulated velocity should lie on it for every damping.

fig, ax = plt.subplots(figsize=(7, 4))
for damping in [0.5, 1.0, 2.0]:
    trajectory = create_simulation(SYSTEM, b=damping).prepare().run(until=8).trajectory_
    times = trajectory.times
    tau = 2 / np.pi * np.arcsin(np.sin(np.pi * times / 2))
    wave = np.sign(np.cos(np.pi * times / 2))
    exact = 0.2 / (2 * np.cosh(damping)) * np.exp(-damping * tau * wave) * wave
    ax.plot(times, np.ravel(trajectory.values('ball.v')), label=f'b = {damping}')
    ax.plot(times, exact, linestyle=':', color='black')
ax.set_xlabel('time')
ax.set_ylabel('velocity v')
ax.set_title('Simulated velocity and the closed form (dotted)')
ax.legend()
plt.show()
