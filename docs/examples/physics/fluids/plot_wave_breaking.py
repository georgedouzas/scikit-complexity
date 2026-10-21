"""
# Breaking of a wave train

A train of deep-water waves whose wave number varies along the train obeys the conservation of waves
`k_t + omega(k)_x = 0` with `omega = sqrt(g k)`. The local wave number travels at the group speed
`C(k) = sqrt(g / k) / 2`, so long waves outrun short ones, the wave number piles up where long waves catch up with
short ones, and the profile of `k` steepens into a jump, a break in the wavelength. The space-time image shows the wave
number, one row per saved step with the earliest at the top, with the smooth modulation turning into a moving jump.
"""

# %%
# The model is created by name from the catalog with 200 finite-volume cells and the wave number
# `k0 (1 + 0.5 sin x)`. The catalog registers the strong-stability-preserving Runge-Kutta stepper for this model, so it
# is passed explicitly. The observer saves every frame of 0.1 with the default blueprint, whose primary view is the
# space-time image of the wave number.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'wave_breaking'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=SSPRK(), dt=0.1, observers=[observer]).run(until=25)

# %%
# Breaking time and the conservation of waves
# -------------------------------------------
# Characteristics of the wave number first cross at `t_b = 1 / max(-C'(k0(x)) k0'(x))`, so the steepest slope of the
# wave number should stay moderate before `t_b` and jump up to the resolution of the grid right after it. The finite
# volumes conserve the total number of waves, the integral of `k`, which should stay at its initial value.

trajectory = simulation.trajectory_
times = trajectory.times
numbers = trajectory['train.k']
spacing = 2 * np.pi / numbers.shape[1]
slope = np.abs(np.diff(numbers, axis=1, append=numbers[:, :1])).max(axis=1) / spacing
grid = np.linspace(0, 2 * np.pi, 2000)
initial = 1 + 0.5 * np.sin(grid)
breaking = 1 / np.max(np.sqrt(1.0) / 4 * initial**-1.5 * 0.5 * np.cos(grid))
drift = np.abs(numbers.sum(axis=1) - numbers[0].sum()) / numbers[0].sum()
fig, (steep, total) = plt.subplots(1, 2, figsize=(10, 4))
steep.semilogy(times, slope, label='simulation')
steep.axvline(breaking, color='gray', linestyle='--', label=f'predicted breaking time {breaking:.1f}')
steep.set_xlabel('time t')
steep.set_ylabel('steepest slope |k_x|')
steep.set_title('The wave number breaks at the predicted time')
steep.legend()
total.semilogy(times[1:], drift[1:] + 1e-17)
total.set_xlabel('time t')
total.set_ylabel('relative change of the total of k')
total.set_title('The number of waves is conserved to round-off')
fig.tight_layout()
plt.show()
