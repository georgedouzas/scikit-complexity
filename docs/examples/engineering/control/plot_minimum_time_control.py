"""
# Minimum-time control

Pontryagin's maximum principle makes the fastest way to bring a linear plant to rest at the origin a bang-bang
control: full push one way, then full push the other, switching once on the curve of the trajectories that end at the
origin. A controller that samples the state at short intervals and holds the control in between keeps the plant in a
small neighbourhood of the origin, whose size shrinks with the interval.
"""

# %%
# The system is created by name from the catalog: a free particle `x'' = u` from `x = 3` with velocity 2, sampled
# every 0.1. The default blueprint shows the position, the velocity, and the control.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'minimum_time_control'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# Three plants
# ------------
# The free particle, the harmonic oscillator `x'' + x = u`, and the saddle `x'' - x = u` each follow arcs of their
# trajectories under `u = 1` and `u = -1` to the switching curve, and then along it to the origin. The dotted line is
# the switching curve.

starts = {'free_particle': (3.0, 2.0), 'harmonic_oscillator': (-2.5, 1.5), 'saddle': (0.5, -0.2)}
grid = {
    'free_particle': np.linspace(-3, 3, 301),
    'harmonic_oscillator': np.linspace(-3, 3, 301),
    'saddle': np.linspace(-1, 1, 201),
}
curves = {
    'free_particle': lambda x: -np.sign(x) * np.sqrt(2 * np.abs(x)),
    'harmonic_oscillator': lambda x: np.where(
        np.abs(x) > 2,
        0,
        np.where(x < 0, np.sqrt(np.abs(1 - (x + 1) ** 2)), -np.sqrt(np.abs(1 - (x - 1) ** 2))),
    ),
    'saddle': lambda x: np.where(x <= 0, np.sqrt(np.abs(x**2 - 2 * x)), -np.sqrt(np.abs(x**2 + 2 * x))),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (plant, (x, v)) in zip(axes, starts.items(), strict=True):
    trajectory = create_simulation(SYSTEM, plant=plant, x=x, v=v, interval=0.01).prepare().run().trajectory_
    ax.plot(np.ravel(trajectory.values('plant.x')), np.ravel(trajectory.values('plant.y')))
    ax.plot(grid[plant], curves[plant](grid[plant]), linestyle=':', color='black')
    ax.set_title(plant.replace('_', ' '))
    ax.set_xlabel('position x')
axes[0].set_ylabel('velocity y')
plt.show()
