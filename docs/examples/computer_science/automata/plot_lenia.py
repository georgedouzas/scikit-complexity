"""
# Lenia

Lenia generalises the Game of Life to continuous states, space, and time: every site holds a value in `[0, 1]`, a
smooth ring kernel replaces the eight neighbours, and a smooth growth map replaces the birth and survival rules. Its
rules grow solitons, localized patterns that keep their shape and move like organisms, such as Orbium unicaudatus.
"""

# %%
# The torus has 64 by 64 sites and holds Orbium, with the kernel radius `R = 13`, `T = 10` steps per unit time, and
# the growth map centred at `mu = 0.15` with width `sigma = 0.015`. The run lasts 200 steps and the image shows the
# first channel.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import LeniaSystem

SYSTEM = 'lenia'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(LeniaSystem(), observers=[observer], random_state=0)

# %%
# A gliding soliton
# -----------------
# Orbium should keep its shape and its mass near 71 while it glides across the torus, wrapping around its edges.

world = np.asarray(trajectory.get_lattice('world.A'))
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, step in zip(axes[:3], (0, 100, 200), strict=True):
    ax.imshow(world[step], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'step {step}')
    ax.set_axis_off()
axes[3].plot(trajectory.times, world.sum(axis=(1, 2)))
axes[3].set_xlabel('step')
axes[3].set_ylabel('mass')
plt.show()

# %%
# The expanded universe
# ---------------------
# Several kernels and channels turn Lenia into a recurrent convolutional network. Two channels that feed each other
# through three kernels, from a random patch, should grow and decay into structures of their own.

rng = np.random.default_rng(0)
initial = np.zeros((2, 64, 64))
initial[:, 22:42, 22:42] = rng.random((2, 20, 20))
kernels = [
    (0, 0, 1.0, 1.0, (1.0,), 0.15, 0.015),
    (1, 0, 0.5, 1.0, (1.0,), 0.2, 0.03),
    (0, 1, 1.0, 1.0, (0.5, 1.0), 0.15, 0.03),
]
expanded = simulate(LeniaSystem(initial=initial, channels=2, kernels=kernels), until=50, random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
for ax, channel in zip(axes, 'AB', strict=True):
    ax.imshow(np.asarray(expanded.get_lattice(f'world.{channel}'))[-1], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'channel {channel} at step 50')
    ax.set_axis_off()
plt.show()
