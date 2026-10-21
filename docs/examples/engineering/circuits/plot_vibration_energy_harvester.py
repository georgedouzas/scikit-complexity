"""
# Vibration energy harvester

The bumps of the road shake the wheel and the body of a car, and a piezoelectric bar between them turns the motion
into a voltage that oscillates with the road. The voltage is largest when the road drives the suspension near one of
its two natural frequencies.
"""

# %%
# The model is created by name from the catalog with a wheel of 40 kg, a body of 400 kg, and a road of amplitude 1 cm
# driving at 9.42 rad/s, and runs for 10 seconds. The default blueprint shows the displacements of both masses with
# their rates and the voltage of the bar.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'vibration_energy_harvester'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=10)

# %%
# Frequency response of the voltage
# ---------------------------------
# Without damping, the two masses have natural frequencies set by the eigenvalues of `M^-1 K`, with the stiffness
# matrix `[[k1 + k2, -k2], [-k2, k2]]`: a slow mode of the body and a fast mode of the wheel. Sweeping the road
# frequency, the steady voltage amplitude should peak near them, the fast one broadened by the damper.

m1, m2, k1, k2 = 40.0, 400.0, 2e5, 2e4
natural = np.sqrt(np.linalg.eigvals(np.diag([1 / m1, 1 / m2]) @ np.array([[k1 + k2, -k2], [-k2, k2]])).real)
frequencies = np.linspace(2.0, 100.0, 50)
amplitudes = []
for omega in frequencies:
    run = Simulation(create(MODEL, omega=omega), dt=0.01).run(until=10).trajectory_
    amplitudes.append(np.abs(run['voltage'][-300:]).max())
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(frequencies, amplitudes, 'o-', markersize=3, label='simulated voltage amplitude')
for value in natural:
    ax.axvline(value, color='black', linestyle='--', linewidth=0.8)
ax.axvline(9.42, color='C3', linewidth=0.8, label='road of the run above')
ax.set_xlabel('road frequency omega (rad/s)')
ax.set_ylabel('voltage amplitude')
ax.set_title('Peaks near the natural frequencies (dashed)')
ax.legend()
plt.show()
