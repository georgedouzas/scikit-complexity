"""
# FitzHugh-Nagumo medium

In an excitable medium of FitzHugh-Nagumo cells a broken wave cannot close: its free end curls round into a spiral
that rotates steadily, sending out wave after wave, the pattern of re-entrant waves in heart tissue.
"""

# %%
# The model is created by name from the catalog on a lattice of 100 by 100 sites, with the upper half excited and its
# left half refractory, and integrated with Runge-Kutta steps of 0.05. The observer saves every frame with the default
# blueprint, whose primary view is the image of the potential.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fitzhugh_nagumo'

system = create(MODEL, shape=(100, 100))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=2.0, solver=RK4(dt=0.05), observers=[observer]).run(until=300)

# %%
# A periodic rotation
# -------------------
# A rotating spiral excites every site once per turn, so the potential at a fixed probe should spike periodically with
# the rotation period of the spiral, and the last snapshot should show the spiral arm.

trajectory = simulation.trajectory_
potential = fields_of(trajectory, 'medium.u')
probe = potential[:, 75, 75]
peaks, _ = find_peaks(probe, height=0.5)
fig, (snapshot, series) = plt.subplots(1, 2, figsize=(11, 4.5))
image = snapshot.imshow(potential[-1], cmap='inferno', origin='lower')
snapshot.plot(75, 75, 'o', color='cyan', label='probe')
snapshot.set_title(f'Potential at t = {trajectory.times[-1]:.0f}')
snapshot.set_xlabel('column')
snapshot.set_ylabel('row')
snapshot.legend()
fig.colorbar(image, ax=snapshot, label='potential u')
series.plot(trajectory.times, probe)
series.plot(trajectory.times[peaks], probe[peaks], 'o', color='black')
series.set_xlabel('time')
series.set_ylabel('potential at the probe')
series.set_title(f'Rotation period {np.diff(trajectory.times[peaks]).mean():.0f}')
fig.tight_layout()
plt.show()
