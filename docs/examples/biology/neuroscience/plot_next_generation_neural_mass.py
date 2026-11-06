"""
# Next-generation neural mass

The exact mean field of excitatory and inhibitory quadratic integrate-and-fire neurons, driven by a slow theta current,
fires bursts of gamma oscillations at every theta cycle, theta-nested gamma oscillations as recorded in the
hippocampus under optogenetic theta stimulation.
"""

# %%
# The system is created by name from the catalog with the PING settings of Segneri et al. (2020) and run for 600 ms
# at a theta frequency of 5 Hz. The observer saves every other millisecond with the default blueprint, whose primary
# view is the time series of the rates and potentials.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'next_generation_neural_mass'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).prepare().run(until=600)

# %%
# Theta-nested gamma
# ------------------
# The spectrum of the mean potential should peak at the theta frequency and again in the gamma band near 40 Hz, while
# the rate shows gamma bursts in every theta cycle.

fine = Simulation(create_system(SYSTEM), dt=0.5).prepare().run(until=2000).trajectory_
potential = fine.values('excitatory.v')[400:]
frequencies, power = welch(potential - potential.mean(), fs=2000.0, nperseg=2048)
fig, (series, spectrum) = plt.subplots(1, 2, figsize=(11, 4.5))
series.plot(fine.times[:2000], 1000 * fine.values('excitatory.r')[:2000], label='excitatory rate')
series.plot(fine.times[:2000], 1000 * fine.values('inhibitory.r')[:2000], label='inhibitory rate')
series.set_xlabel('time (ms)')
series.set_ylabel('rate (Hz)')
series.set_title('Gamma bursts in every theta cycle')
series.legend()
spectrum.semilogy(frequencies, power)
spectrum.set_xlim(0, 100)
spectrum.set_xlabel('frequency (Hz)')
spectrum.set_ylabel('power of the mean potential')
spectrum.set_title(f'Gamma peak near {frequencies[np.argmax(power * (frequencies > 20))]:.0f} Hz')
fig.tight_layout()
plt.show()
