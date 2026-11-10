"""
# Real quadratic automaton

The cells sit on a ring of 64 cells over the reals, and every cell becomes `x(n + 1) - x(n)^2`. The automaton is
pre-injective, so two configurations that differ on finitely many cells have different images, but it is not
injective: the constant configurations 0 and 1 both map to 0. Its image is dense without being closed or everything.
"""

# %%
# The run starts from small random values, which the square pulls down at every step.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import RealQuadraticAutomatonSystem

SYSTEM = 'real_quadratic_automaton'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(RealQuadraticAutomatonSystem(size=64, amplitude=0.25, random_state=0), observers=[observer])

# %%
# Collisions and the drift
# ------------------------
# The constant configurations 0 and 1 should have the same image 0, the failure of injectivity. Changing one cell of a
# configuration should change its image at that cell and the cell before it, so finitely supported changes never
# vanish, the pre-injectivity. The mean of the cells drifts down by the mean square at every step.


def step(values):
    return simulate(RealQuadraticAutomatonSystem(size=values.size, initial=values), until=1)


zero = step(np.zeros(8)).get_values('cells.value')[-1]
one = step(np.ones(8)).get_values('cells.value')[-1]
start = trajectory.get_values('cells.value')[0]
changed = start.copy()
changed[20] += 0.1
moved = np.flatnonzero(step(changed).get_values('cells.value')[-1] != step(start).get_values('cells.value')[-1])
values = trajectory.get_values('cells.value')
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, values.mean(axis=1), 'o-', label='mean of the cells')
ax.plot(trajectory.times, np.median(values, axis=1), 's-', label='median of the cells')
ax.set_xlabel('step')
ax.set_title(f'tau(0) = tau(1): {np.array_equal(zero, one)}, a change at cell 20 moves cells {moved.tolist()}')
ax.legend()
plt.show()
