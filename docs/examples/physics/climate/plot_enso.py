"""
# Delayed oscillator of El Niño

The sea surface temperature of the eastern Pacific grows by local feedback until the delayed return of reflected ocean
waves turns it around, so it swings regularly between El Niño, a warm anomaly, and La Niña, a cold one.
"""

# %%
# The system is created by name from the catalog with the delayed feedback `alpha = 0.75` and the delay of six time
# units, starting from a small warm anomaly. The default blueprint shows the anomaly in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import spectrum
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'enso'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=200)

# %%
# A limit cycle with one period
# -----------------------------
# Plotted against its delayed value, the anomaly should trace one closed loop once the oscillation is established, and
# its power spectrum should have a sharp peak at the period of the oscillation, nearly three times the delay, with
# harmonics at its multiples.

trajectory = simulation.trajectory_
anomaly = trajectory.values('pacific.T')
shift = int(6.0 / 0.5)
settled = slice(len(anomaly) // 2, None)
power = spectrum(anomaly[settled], dt=0.5)
frequencies, density = power.frequencies, power.power[0]
peak = frequencies[1:][np.argmax(density[1:])]
fig, (loop, spectrum_axes) = plt.subplots(1, 2, figsize=(10, 4))
loop.plot(anomaly[settled][shift:], anomaly[settled][:-shift])
loop.set_xlabel('anomaly T(t)')
loop.set_ylabel('delayed anomaly T(t - delta)')
loop.set_title('One closed loop')
spectrum_axes.semilogy(frequencies[1:], density[1:])
spectrum_axes.axvline(peak, color='black', linestyle='--', label=f'period {1 / peak:.1f}')
spectrum_axes.set_xlabel('frequency')
spectrum_axes.set_ylabel('power spectral density')
spectrum_axes.set_title('Power spectrum of the anomaly')
spectrum_axes.legend()
fig.tight_layout()
plt.show()
