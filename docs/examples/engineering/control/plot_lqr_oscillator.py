"""
# Linear-quadratic regulated oscillator

A harmonic oscillator released from a displacement of 1 would swing forever. A linear-quadratic regulator, whose gain
comes from the Riccati equation of the oscillator's linearisation, pushes against the motion so that the oscillator
comes to rest in a few seconds.
"""

# %%
# The system is built from its class with the natural frequency 1, the weight 1 of the displacement and the
# velocity, and the weight 0.1 of the force, so the gain is `K = (2.32, 3.82)`. RK4 integrates it with steps of 1 ms,
# and the regulator sets the force after every step. The default blueprint shows the displacement and the force in one
# time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import linearization
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.control import LqrOscillatorSystem

SYSTEM = 'lqr_oscillator'

system = LqrOscillatorSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(system, observers=[observer])

# %%
# The decay rate of the closed loop
# ---------------------------------
# The closed loop `x' = (A - B K) x` decays at the rate of its slowest eigenvalue, so the energy, a quadratic form of
# the state, should fall as `exp(2 Re(lambda) t)` once the faster mode has died out.

parts = system.prepare().system_
linear = linearization(System(elements=list(parts.elements)))
gain = np.array([parts.interactions[0].gain])
slowest = max(np.linalg.eigvals(linear.A - linear.B @ gain).real)
times, energy = trajectory.times, trajectory.values('energy')
decay = energy[40] * np.exp(2 * slowest * (times - times[40]))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(times, energy, label='energy of the oscillator')
ax.semilogy(times, decay, '--', label=f'exp(2 t Re lambda), Re lambda = {slowest:.2f}')
ax.set_xlabel('time')
ax.set_ylabel('energy (x\'^2 + omega^2 x^2) / 2')
ax.set_title('The regulator removes the energy at the closed-loop rate')
ax.legend()
plt.show()
