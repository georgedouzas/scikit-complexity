"""
# RLC circuit

The charge on the capacitor of a series RLC circuit rings at the natural frequency `1 / sqrt(L C)`, and the resistor
damps the ringing so the oscillation decays exponentially.
"""

# %%
# The model is created by name from the catalog with `L = 1`, `R = 0.1`, `C = 1`, no source, and the capacitor charged
# to `q = 1`, and runs for 50 time units. The default blueprint shows the charge and its rate, the current, in one time
# series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rlc_circuit'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# Damped ringing and resonance
# ----------------------------
# Underdamped, the charge should follow `exp(-R t / (2 L)) cos(omega_d t)` with `omega_d^2 = 1 / (L C) - R^2 / (4 L^2)`.
# Driven by `V cos(omega t)`, the steady amplitude should follow the resonance curve
# `V / sqrt((1 / C - L omega^2)^2 + (R omega)^2)`, peaked at the natural frequency.

L, R, C = 1.0, 0.1, 1.0
trajectory = simulation.trajectory_
times = trajectory.times
omega_d = np.sqrt(1 / (L * C) - R**2 / (4 * L**2))
exact = np.exp(-R * times / (2 * L)) * (np.cos(omega_d * times) + R / (2 * L * omega_d) * np.sin(omega_d * times))
frequencies = np.linspace(0.5, 1.5, 21)
amplitudes = []
for omega in frequencies:
    driven = Simulation(create(MODEL, q=0.0, V=0.1, omega=omega), dt=0.1).run(until=150).trajectory_
    amplitudes.append(np.abs(driven['circuit.q'][-300:]).max())
grid = np.linspace(0.5, 1.5, 300)
fig, (ringing, resonance) = plt.subplots(1, 2, figsize=(10, 4))
ringing.plot(times, trajectory['circuit.q'], label='simulation')
ringing.plot(times, exact, color='black', linestyle='--', linewidth=1, label='exact solution')
ringing.plot(times, np.exp(-R * times / (2 * L)), color='gray', linewidth=0.8, label='envelope exp(-R t / 2L)')
ringing.set_xlabel('time')
ringing.set_ylabel('charge q')
ringing.set_title('Damped ringing')
ringing.legend()
resonance.plot(grid, 0.1 / np.sqrt((1 / C - L * grid**2) ** 2 + (R * grid) ** 2), color='black', label='theory')
resonance.plot(frequencies, amplitudes, 'o', label='simulation')
resonance.set_xlabel('driving frequency omega')
resonance.set_ylabel('steady amplitude of q')
resonance.set_title('Resonance at 1 / sqrt(L C)')
resonance.legend()
fig.tight_layout()
plt.show()
