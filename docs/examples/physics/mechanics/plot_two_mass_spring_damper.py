"""
# Two-mass spring-damper

Two masses tied by springs to two walls and to each other move as a mix of two normal modes: swinging together at the
slow frequency, and against each other at the fast one, so each mass alone traces a beating, quasi-periodic curve.
"""

# %%
# With equal masses and springs, `m = k = 1`, the in-phase mode has the frequency 1 and the anti-phase mode `sqrt(3)`.
# Both masses start displaced from rest at `x = 1` and `x = 2`, the first by 0.2 and the second not at all, which
# excites both modes. The default blueprint shows the position of the first mass first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import power_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'two_mass_spring_damper'

system = create(MODEL, x1=1.2, x2=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Normal-mode spectrum
# --------------------
# The power spectrum of a long run of the first mass should show two sharp peaks, at the angular frequencies 1 and
# `sqrt(3)` of the normal modes, dashed, and nothing else since the system is linear and undamped.

run = Simulation(create(MODEL, x1=1.2, x2=2.0), dt=0.1).run(until=2000)
frequencies, density = power_spectrum(np.ravel(run.trajectory_['first.x']) - 1.0, dt=0.1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(2 * np.pi * frequencies, density)
for omega in (1.0, np.sqrt(3.0)):
    ax.axvline(omega, linestyle='--', color='black')
ax.set_xlim(0, 4)
ax.set_xlabel('angular frequency ω')
ax.set_ylabel('power spectral density')
ax.set_title('Peaks at the normal modes 1 and sqrt(3)')
plt.show()
