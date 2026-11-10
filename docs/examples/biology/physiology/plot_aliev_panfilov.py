"""
# Aliev-Panfilov

A broken wave of excitation in cardiac tissue curls around its free end into a rotating spiral. The spiral keeps
re-exciting the tissue it has left once it has recovered, a re-entrant arrhythmia that drives the heart far
faster than its pacemaker, a tachycardia.
"""

# %%
# The example builds `AlievPanfilovSystem` on a lattice of 100 by 100 sites, with the upper half excited and its left
# half refractory, and integrates it with Runge-Kutta steps of 0.05. The observer saves every frame with the default
# blueprint, whose primary view is the image of the potential.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import AlievPanfilovSystem

SYSTEM = 'aliev_panfilov'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(AlievPanfilovSystem(shape=(100, 100)), solver=RK4Solver(dt=0.05), observers=[observer], until=300)

# %%
# A periodic rotation
# -------------------
# A rotating spiral excites every site once per turn, so the potential at a fixed probe should spike periodically with
# the rotation period of the spiral, and the last snapshot should show the spiral arm.

potential = trajectory.get_lattice('tissue.u')
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
