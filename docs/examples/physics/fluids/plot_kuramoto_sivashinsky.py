"""
# Kuramoto-Sivashinsky equation

The Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx` describes flame fronts and thin falling films. The
second derivative pumps energy into long waves, the fourth derivative damps short ones, and the nonlinear term
carries energy between scales, so a smooth start breaks up into cells of a typical size that split, merge, and drift
without ever settling. The space-time image shows the field on a periodic line of length `32 pi`, one row per saved
step with the earliest at the top, a chaotic pattern of ridges that split, merge, and drift.
"""

# %%
# The model is created by name from the catalog with 128 Fourier modes, started from Kassam and Trefethen's
# `cos(x / 16) (1 + sin(x / 16))`. The catalog registers the fourth-order exponential time differencing stepper for
# this stiff equation, so it is passed explicitly with steps of 0.25. The observer saves every frame of 0.5 with the
# default blueprint, whose primary view is the space-time image of the field.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ETDRK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kuramoto_sivashinsky'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=ETDRK4(dt=0.25), dt=0.5, observers=[observer]).run(until=149.5)

# %%
# The cell size set by the linear instability
# -------------------------------------------
# A Fourier mode `exp(i q x)` grows at the linear rate `q^2 - q^4`, positive for `q < 1` and largest at
# `q = 1 / sqrt 2`. In the chaotic state the time-averaged power spectrum of a long run should peak near this
# wavenumber, set the cell size near `2 pi sqrt 2 = 8.9`, and fall off exponentially above `q = 1`, where the modes
# decay.

run = Simulation(create(MODEL), solver=ETDRK4(dt=0.25), dt=1.0).run(until=1000)
field = run.trajectory_['line.u'][200:]
length = 32 * np.pi
wavenumbers = 2 * np.pi * np.fft.rfftfreq(field.shape[1], d=length / field.shape[1])
power = np.mean(np.abs(np.fft.rfft(field, axis=1)) ** 2, axis=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(wavenumbers[1:], power[1:], 'o-', markersize=3, label='time-averaged power spectrum')
ax.axvline(1 / np.sqrt(2), color='black', linestyle='--', label='fastest growing q = 1 / sqrt 2')
ax.axvline(1.0, color='gray', linestyle=':', label='stable above q = 1')
ax.set_xlim(0, 2.6)
ax.set_xlabel('wavenumber q')
ax.set_ylabel('power')
ax.set_title('Chaos selects the scale of the linear instability')
ax.legend()
fig.tight_layout()
plt.show()
