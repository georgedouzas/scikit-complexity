"""
# Saltzman-Maasch glacial cycles

Ice volume, carbon dioxide, and the deep ocean temperature drive each other around a self-sustained cycle of about
100 kyr, without any orbital forcing: the ice builds up while carbon dioxide falls, then retreats as it rises.
"""

# %%
# The system is built from its class with the default parameters and no forcing, and runs for 50 time units
# of 10 kyr, half a million years. The default blueprint shows the three anomalies in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import SaltzmanGlaciationSystem

SYSTEM = 'saltzman_glaciation'

system = SaltzmanGlaciationSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer])

# %%
# The 100 kyr cycle
# -----------------
# On the limit cycle ice volume and carbon dioxide should trace one closed loop, and the power spectrum of a long run of
# ice volume should peak at a period near 100 kyr.

run = simulate(SaltzmanGlaciationSystem(), until=400)
settled = run.times > 20
ice, carbon = run.values('climate.x')[settled], run.values('climate.y')[settled]
power = spectrum(ice, dt=0.05)
frequencies, density = power.frequencies, power.power[0]
peak = frequencies[1:][np.argmax(density[1:])]
fig, (loop, spectrum_axes) = plt.subplots(1, 2, figsize=(10, 4))
loop.plot(ice, carbon)
loop.set_xlabel('ice volume x')
loop.set_ylabel('carbon dioxide y')
loop.set_title('One closed loop')
spectrum_axes.semilogy(10 / frequencies[1:], density[1:])
spectrum_axes.axvline(10 / peak, color='black', linestyle='--', label=f'period {10 / peak:.0f} kyr')
spectrum_axes.set_xscale('log')
spectrum_axes.set_xlabel('period (kyr)')
spectrum_axes.set_ylabel('power spectral density')
spectrum_axes.set_title('Power spectrum of ice volume')
spectrum_axes.legend()
fig.tight_layout()
plt.show()
