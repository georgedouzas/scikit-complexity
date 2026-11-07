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

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.nonlinear_oscillators import ForcedVanDerPolSystem

SYSTEM = 'forced_van_der_pol'

system = ForcedVanDerPolSystem(x=2.0, v=0.0, mu=1.0, A=0.3, omega=0.95)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
trajectory = simulate(system, until=60, observers=[observer])

# %%
# Frequency locking
# -----------------
# The frequency of the response is measured from the upward zero crossings of `x` after a transient. Inside the
# locking range, around the natural frequency of about 0.94, the ratio of the response frequency to the driving
# frequency should be exactly 1, a flat plateau, while outside it the oscillator should drift at a frequency closer to
# its own, so the ratio departs from 1 on both sides.


def response_frequency(omega):
    """Return the mean angular frequency of the response after a transient."""
    run = simulate(ForcedVanDerPolSystem(x=2.0, v=0.0, mu=1.0, A=0.3, omega=omega), until=400, dt=0.025)
    times, x = np.asarray(run.times), np.ravel(run.values('oscillator.x'))
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
