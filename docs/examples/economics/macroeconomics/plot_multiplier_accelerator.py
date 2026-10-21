"""
# Multiplier-accelerator

Samuelson's business cycle: consumption follows last period's income through the multiplier, and investment follows
the last change of consumption through the accelerator. The interplay alone makes income oscillate around its
equilibrium `G / (1 - c)`, and the propensity to consume `c` and the accelerator `v` decide whether the cycles die out
or explode.
"""

# %%
# With the propensity to consume `c = 0.8`, the accelerator `v = 0.9`, and government spending `G = 20`, income starts
# just above its equilibrium of 100 and swings around it in damped cycles, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'multiplier_accelerator'
G, UNTIL = 20.0, 40

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, income=110.0, previous=100.0, c=0.8, v=0.9, G=G)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=UNTIL)

# %%
# Samuelson's regions
# -------------------
# The deviation from equilibrium follows `x' = c (1 + v) x - c v x_prev`. Its characteristic roots are complex, and
# income oscillates, below the curve `c (1 + v)^2 = 4 v`, and they leave the unit circle, and the motion grows, above
# `c v = 1`. One run in each of the four regions should behave as its region predicts, and the period of
# the oscillations should be `2 pi / arg(lambda)` of the characteristic root `lambda`.

cases = {(0.6, 0.05): 'monotone, damped', (0.8, 0.9): 'oscillating, damped', (0.8, 1.5): 'oscillating, explosive'}
cases[(0.9, 4.0)] = 'monotone, explosive'
fig, (paths, regions) = plt.subplots(1, 2, figsize=(10, 4))
for index, ((c, v), name) in enumerate(cases.items()):
    trajectory = Simulation(create(MODEL, income=110.0, previous=100.0, c=c, v=v, G=G), dt=1).run(until=25).trajectory_
    deviation = trajectory['economy.Y'] - G / (1 - c)
    root = np.roots([1, -c * (1 + v), c * v])[0]
    label = f'{name}, period {2 * np.pi / abs(np.angle(root)):.1f}' if np.iscomplex(root) else name
    paths.plot(trajectory.times, deviation, 'o-', ms=3, color=f'C{index}', label=label)
    regions.plot(v, c, 'o', color=f'C{index}', ms=8)
paths.set_yscale('symlog', linthresh=1)
paths.set_xlabel('period t')
paths.set_ylabel('income minus G / (1 - c)')
paths.set_title('Four kinds of cycle')
paths.legend(fontsize=8)
accelerators = np.linspace(0.01, 5, 400)
regions.plot(accelerators, 4 * accelerators / (1 + accelerators) ** 2, 'k', label='c (1 + v)^2 = 4 v')
regions.plot(accelerators, np.minimum(1 / accelerators, 1), 'k--', label='c v = 1')
regions.set_ylim(0, 1)
regions.set_xlabel('accelerator v')
regions.set_ylabel('propensity to consume c')
regions.set_title('Regions of the (v, c) plane')
regions.legend(loc='lower right')
fig.tight_layout()
plt.show()
