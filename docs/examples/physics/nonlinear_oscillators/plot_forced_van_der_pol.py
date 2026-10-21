"""
# Forced Van der Pol

A self-sustained oscillator pushed by a periodic force gives up its own rhythm: near its natural frequency it locks to
the drive and oscillates exactly at the driving frequency, over a whole plateau of driving frequencies.
"""

# %%
# The oscillator with `μ = 1` is driven with amplitude `A = 0.3` at the frequency `ω = 0.95`, inside the locking
# range. The default blueprint shows the position and the velocity, which settle on a periodic response at the drive
# period.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'forced_van_der_pol'

system = create(MODEL, x=2.0, v=0.0, mu=1.0, A=0.3, omega=0.95)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=60)

# %%
# Frequency locking
# -----------------
# The frequency of the response is measured from the upward zero crossings of `x` after a transient. Inside the
# locking range, around the natural frequency of about 0.94, the ratio of the response frequency to the driving
# frequency should be exactly 1, a flat plateau, while outside it the oscillator should drift at a frequency closer to
# its own, so the ratio departs from 1 on both sides.


def response_frequency(omega):
    """Return the mean angular frequency of the response after a transient."""
    run = Simulation(create(MODEL, x=2.0, v=0.0, mu=1.0, A=0.3, omega=omega), dt=0.025).run(until=400)
    times, x = np.asarray(run.trajectory_.times), np.ravel(run.trajectory_['oscillator.x'])
    late = times > 150
    crossings = times[late][1:][(x[late][:-1] < 0) & (x[late][1:] >= 0)]
    return 2 * np.pi * (len(crossings) - 1) / (crossings[-1] - crossings[0])


omegas = np.linspace(0.6, 1.4, 21)
ratios = [response_frequency(omega) / omega for omega in omegas]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(omegas, ratios, 'o-', markersize=4)
ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('driving frequency ω')
ax.set_ylabel('response frequency / ω')
ax.set_title('Locking plateau of the forced Van der Pol oscillator')
plt.show()
