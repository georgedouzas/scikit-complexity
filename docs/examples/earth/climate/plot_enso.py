"""
# Delayed oscillator of El Niño

The sea surface temperature of the eastern Pacific grows by local feedback until the delayed return of reflected ocean
waves turns it around, so it swings regularly between El Niño, a warm anomaly, and La Niña, a cold one.
"""

# %%
# The model is created by name from the catalog with the delayed feedback `alpha = 0.75` and the delay of six time
# units, starting from a small warm anomaly. The default blueprint shows the anomaly in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import power_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'enso'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=200)

# %%
# A limit cycle with one period
# -----------------------------
# Plotted against its delayed value, the anomaly should trace one closed loop once the oscillation is established, and
# its power spectrum should have a sharp peak at the period of the oscillation, nearly three times the delay, with
# harmonics at its multiples.

trajectory = simulation.trajectory_
anomaly = trajectory['pacific.T']
shift = int(6.0 / 0.5)
settled = slice(len(anomaly) // 2, None)
frequencies, density = power_spectrum(anomaly[settled], dt=0.5)
peak = frequencies[1:][np.argmax(density[1:])]
fig, (loop, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
loop.plot(anomaly[settled][shift:], anomaly[settled][:-shift])
loop.set_xlabel('anomaly T(t)')
loop.set_ylabel('delayed anomaly T(t - delta)')
loop.set_title('One closed loop')
spectrum.semilogy(frequencies[1:], density[1:])
spectrum.axvline(peak, color='black', linestyle='--', label=f'period {1 / peak:.1f}')
spectrum.set_xlabel('frequency')
spectrum.set_ylabel('power spectral density')
spectrum.set_title('Power spectrum of the anomaly')
spectrum.legend()
fig.tight_layout()
plt.show()
