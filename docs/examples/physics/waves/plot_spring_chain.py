"""
# Spring chains and guitar string

Fifteen masses joined by springs between two fixed anchors form a discrete string. Plucked at its middle into a
triangle, the chain does not keep the triangle: its sharp corner splits into kinks that run to the ends and back, the
sum of the string's normal modes beating together.
"""

# %%
# The middle mass is pulled up by 0.05 of the length and the chain released at rest. The default blueprint draws the
# masses in the plane of the string, with the short trails of their recent positions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'spring_chain'
N_MASSES = 15

system = create(MODEL, n_masses=N_MASSES, pluck=0.05)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=4)

# %%
# Normal modes
# ------------
# Springs of zero rest length make the transverse motion linear, so each normal mode `sin(n π i / (N + 1))` keeps its
# own amplitude and oscillates at `2 sqrt(k / m) sin(n π / (2 (N + 1)))`. The projections of the heights on the first
# three odd modes, which the symmetric pluck excites, should follow cosines at those frequencies.

trajectory = simulation.trajectory_
heights = np.stack([np.asarray(trajectory[f'mass_{index}.x'])[:, 1] for index in range(N_MASSES)], axis=1)
sites = np.arange(1, N_MASSES + 1)
times = np.asarray(trajectory.times)
fig, ax = plt.subplots(figsize=(7, 4))
for number in (1, 3, 5):
    mode = np.sin(number * np.pi * sites / (N_MASSES + 1))
    amplitude = heights @ mode / (mode @ mode)
    frequency = 2 * np.sqrt(1.0 / 0.01) * np.sin(number * np.pi / (2 * (N_MASSES + 1)))
    line = ax.plot(times, amplitude, label=f'mode {number}')[0]
    ax.plot(times, amplitude[0] * np.cos(frequency * times), '--', color=line.get_color(), linewidth=0.8)
ax.set_xlabel('time')
ax.set_ylabel('mode amplitude')
ax.set_title('Every mode oscillates at its own frequency (dashed)')
ax.legend()
plt.show()
