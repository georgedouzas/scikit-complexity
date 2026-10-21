"""
# Saltzman-Maasch glacial cycles

Ice volume, carbon dioxide, and the deep ocean temperature drive each other around a self-sustained cycle of about
100 kyr, without any orbital forcing: the ice builds up while carbon dioxide falls, then retreats as it rises.
"""

# %%
# The model is created by name from the catalog with the default parameters and no forcing, and runs for 50 time units
# of 10 kyr, half a million years. The default blueprint shows the three anomalies in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import power_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'saltzman_glaciation'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# The 100 kyr cycle
# -----------------
# On the limit cycle ice volume and carbon dioxide should trace one closed loop, and the power spectrum of a long run of
# ice volume should peak at a period near 100 kyr.

run = Simulation(create(MODEL), dt=0.05).run(until=400).trajectory_
settled = run.times > 20
ice, carbon = run['climate.x'][settled], run['climate.y'][settled]
frequencies, density = power_spectrum(ice, dt=0.05)
peak = frequencies[1:][np.argmax(density[1:])]
fig, (loop, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
loop.plot(ice, carbon)
loop.set_xlabel('ice volume x')
loop.set_ylabel('carbon dioxide y')
loop.set_title('One closed loop')
spectrum.semilogy(10 / frequencies[1:], density[1:])
spectrum.axvline(10 / peak, color='black', linestyle='--', label=f'period {10 / peak:.0f} kyr')
spectrum.set_xscale('log')
spectrum.set_xlabel('period (kyr)')
spectrum.set_ylabel('power spectral density')
spectrum.set_title('Power spectrum of ice volume')
spectrum.legend()
fig.tight_layout()
plt.show()
