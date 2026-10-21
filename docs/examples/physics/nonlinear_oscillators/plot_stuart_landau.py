"""
# Stuart-Landau

Past a Hopf bifurcation the rest state is unstable: a small perturbation grows into a sinusoidal oscillation whose
amplitude saturates at `sqrt(μ)`, the universal behaviour near the onset of any oscillation.
"""

# %%
# The oscillator starts with amplitude 0.01, with `μ = 1`, `ω = 2`, and `β = 1`. The default blueprint shows the real
# and imaginary parts of `z`, which swell from almost nothing to steady oscillations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stuart_landau'

mu, omega, beta = 1.0, 2.0, 1.0
system = create(MODEL, x=0.01, y=0.0, mu=mu, omega=omega, beta=beta)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=20)

# %%
# Amplitude and frequency
# -----------------------
# The amplitude `r = |z|` obeys `r' = μ r - r^3`, solved by `r(t)^2 = μ / (1 + (μ / r0^2 - 1) exp(-2 μ t))`, and on
# the cycle the phase turns at the frequency `ω - β μ`. The simulated amplitude should lie on the exact curve, and
# the phase should advance at the predicted rate once the amplitude has saturated.

times = np.asarray(simulation.trajectory_.times)
z = np.ravel(simulation.trajectory_['oscillator.x']) + 1j * np.ravel(simulation.trajectory_['oscillator.y'])
exact = np.sqrt(mu / (1 + (mu / 0.01**2 - 1) * np.exp(-2 * mu * times)))
phase = np.unwrap(np.angle(z))
fig, (amplitude, rotation) = plt.subplots(1, 2, figsize=(10, 4))
amplitude.plot(times, np.abs(z), label='simulated')
amplitude.plot(times, exact, '--', color='black', label='exact')
amplitude.set_xlabel('time')
amplitude.set_ylabel('|z|')
amplitude.set_title('Growth and saturation at sqrt(μ)')
amplitude.legend()
rotation.plot(times, phase, label='simulated')
rotation.plot(times, phase[-1] + (omega - beta * mu) * (times - times[-1]), '--', color='black', label='ω - β μ')
rotation.set_xlabel('time')
rotation.set_ylabel('unwrapped phase of z')
rotation.set_title('Frequency shifted by the nonlinearity')
rotation.legend()
fig.tight_layout()
plt.show()
