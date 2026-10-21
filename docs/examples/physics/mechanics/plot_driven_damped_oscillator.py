"""
# Driven damped oscillator

A weakly damped mass on a spring is pushed at its natural frequency: the transient dies away and the oscillation grows
to a steady amplitude far larger than the push, the resonance of the response curve.
"""

# %%
# The mass starts at rest and the force `F = 0.5` drives it at its natural frequency `ω = 1` with the damping
# `c = 0.1`. The default blueprint shows the displacement swelling to its steady amplitude `F / (c ω) = 5`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'driven_damped_oscillator'

system = create(MODEL, x=0.0, v=0.0, c=0.1, F=0.5, omega=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=90)

# %%
# Resonance curve
# ---------------
# After the transient the response has the driving frequency and the amplitude
# `F / sqrt((k - m ω^2)^2 + (c ω)^2)`. The steady amplitudes measured over a sweep of the driving frequency should lie
# on that curve, with its sharp peak at the natural frequency.

frequencies = np.linspace(0.5, 1.5, 21)
amplitudes = []
for frequency in frequencies:
    run = Simulation(create(MODEL, x=0.0, v=0.0, c=0.1, F=0.5, omega=frequency), dt=0.05).run(until=150)
    steady = np.ravel(run.trajectory_['mass.x'])[np.asarray(run.trajectory_.times) > 100]
    amplitudes.append(np.abs(steady).max())
grid = np.linspace(0.5, 1.5, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, 0.5 / np.sqrt((1 - grid**2) ** 2 + (0.1 * grid) ** 2), color='black', label='theory')
ax.plot(frequencies, amplitudes, 'o', label='simulated')
ax.set_xlabel('driving frequency ω')
ax.set_ylabel('steady amplitude')
ax.set_title('Resonance of the driven damped oscillator')
ax.legend()
plt.show()
