"""
# Linear-quadratic regulated oscillator

A harmonic oscillator released from a displacement of 1 would swing forever. A linear-quadratic regulator, whose gain
comes from the Riccati equation of the oscillator's linearisation, pushes against the motion so that the oscillator
comes to rest in a few seconds.
"""

# %%
# The model is created by name from the catalog with the natural frequency 1, the weight 1 of the displacement and the
# velocity, and the weight 0.1 of the force, so the gain is `K = (2.32, 3.82)`. RK4 integrates it with steps of 1 ms,
# and the regulator sets the force after every step. The default blueprint shows the displacement and the force in one
# time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import linearize
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'lqr_oscillator'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(
    system,
    dt=0.05,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=10)

# %%
# The decay rate of the closed loop
# ---------------------------------
# The closed loop `x' = (A - B K) x` decays at the rate of its slowest eigenvalue, so the energy, a quadratic form of
# the state, should fall as `exp(2 Re(lambda) t)` once the faster mode has died out.

model = linearize(System(components=list(system.components)))
gain = np.array([system.interactions[0].gain])
slowest = max(np.linalg.eigvals(model.A - model.B @ gain).real)
trajectory = simulation.trajectory_
times, energy = trajectory.times, trajectory['energy']
decay = energy[40] * np.exp(2 * slowest * (times - times[40]))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(times, energy, label='energy of the oscillator')
ax.semilogy(times, decay, '--', label=f'exp(2 t Re lambda), Re lambda = {slowest:.2f}')
ax.set_xlabel('time')
ax.set_ylabel('energy (x\'^2 + omega^2 x^2) / 2')
ax.set_title('The regulator removes the energy at the closed-loop rate')
ax.legend()
plt.show()
